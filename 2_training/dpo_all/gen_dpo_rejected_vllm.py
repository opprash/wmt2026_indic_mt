# -*- coding: utf-8 -*-
"""
DPO preference-data generator (rejected-sample mining).

For each training pair, the post-SFT model samples a `rejected` response
(temperature starts at 1.2 and steps up until rejected != chosen). Outputs
ShareGPT-DPO format (conversations / chosen / rejected) for LLaMA-Factory DPO;
run fix_dpo_data.py afterwards to normalize the `from` field to "gpt".

Multi-GPU sharded HF generate. Edit MODEL_NAME / INPUT_FILE / OUTPUT_FILE below.
(Referenced from the original train/predict_vllm_full.py.)
"""
import os

GPU_IDS = [4, 5, 6, 7]

# 父进程兜底：限制可见卡
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in GPU_IDS)

import json
import queue
import torch.multiprocessing as mp
from tqdm import tqdm

# ========== 配置区 ==========
# 全参微调后的模型路径（直接是可加载的 HF 格式权重目录）
MODEL_NAME      = "/base/rd1/large_models/train_save/full/wmt_total/hy_bodo/3e-5"
INPUT_FILE      = "./data/bodo_train.json"
OUTPUT_FILE     = "./data/hy_bodo_dpo.json"

BATCH_SIZE      = 64        # HF generate，按显存调
TEMPERATURE     = 1.2
TEMP_STEP       = 0.1
MAX_NEW_TOKENS  = 512
MAX_INPUT_LEN   = 512
# ============================


def process_shard(shard_index, gpu_id, shard, result_queue):
    # 每个子进程只看到一张物理卡（在 torch 任何 CUDA 操作之前生效）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[GPU {gpu_id}] 加载模型 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[GPU {gpu_id}] 模型加载完成")

    def build_prompt(instruction):
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def hf_generate(text_list, temperature):
        model_inputs = tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            return_token_type_ids=False,   # HunYuan 的 generate 不接受这个 kwarg
        ).to("cuda:0")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = model_inputs.input_ids.shape[1]
        new_ids   = generated_ids[:, input_len:]
        responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]

    dpo_data = []
    for batch_start in tqdm(
        range(0, len(shard), BATCH_SIZE),
        desc=f"GPU {gpu_id}",
        position=shard_index,
    ):
        batch   = shard[batch_start: batch_start + BATCH_SIZE]
        prompts = [build_prompt(item["instruction"]) for item in batch]
        chosens = [item["output"].strip() for item in batch]

        batch_rejected  = [None] * len(batch)
        pending_indices = list(range(len(batch)))

        attempt = 0
        while pending_indices:
            temperature     = TEMPERATURE + TEMP_STEP * attempt
            pending_prompts = [prompts[i] for i in pending_indices]
            responses       = hf_generate(pending_prompts, temperature)

            still_pending = []
            for idx, resp in zip(pending_indices, responses):
                if resp != chosens[idx]:
                    batch_rejected[idx] = resp
                else:
                    still_pending.append(idx)
            pending_indices = still_pending
            attempt += 1
            if pending_indices:
                print(f"[GPU {gpu_id}] 第 {attempt} 轮仍有 {len(pending_indices)} 条未达成，"
                      f"temperature 升至 {temperature + TEMP_STEP:.2f}")

        for i, item in enumerate(batch):
            dpo_data.append({
                "conversations": [{"from": "human", "value": item["instruction"]}],
                "chosen":        {"from": "gpt", "value": chosens[i]},
                "rejected":      {"from": "gpt",   "value": batch_rejected[i]},
            })

    print(f"[GPU {gpu_id}] 完成，生成 {len(dpo_data)} 条")
    result_queue.put((shard_index, dpo_data))


def split_data(data, n):
    size = (len(data) + n - 1) // n
    return [data[i * size: (i + 1) * size] for i in range(n)]


def main():
    print(f"读取数据：{INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"共 {len(data)} 条数据，分配到 GPU {GPU_IDS}")

    shards = split_data(data, len(GPU_IDS))
    for gpu_id, shard in zip(GPU_IDS, shards):
        print(f"  GPU {gpu_id}: {len(shard)} 条")

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    processes = []

    for shard_index, (gpu_id, shard) in enumerate(zip(GPU_IDS, shards)):
        p = mp.Process(
            target=process_shard,
            args=(shard_index, gpu_id, shard, result_queue),
        )
        p.start()
        processes.append(p)

    # 先 get 再 join，并轮询子进程死活，避免大对象死锁 & 异常静默
    results = {}
    while len(results) < len(GPU_IDS):
        try:
            shard_index, dpo_data = result_queue.get(timeout=5)
            results[shard_index] = dpo_data
        except queue.Empty:
            dead = [(i, p.exitcode) for i, p in enumerate(processes)
                    if not p.is_alive() and p.exitcode != 0]
            if dead:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError(f"子进程异常退出（shard_index, exitcode）：{dead}")

    for p in processes:
        p.join()

    all_dpo_data = []
    for i in range(len(GPU_IDS)):
        all_dpo_data.extend(results.get(i, []))

    print(f"\n✅ 全部完成！共生成 {len(all_dpo_data)} 条DPO数据")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_dpo_data, f, ensure_ascii=False, indent=4)
    print(f"已写入：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
