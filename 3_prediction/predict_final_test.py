# -*- coding: utf-8 -*-
import os
import json
import queue
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path

GPU_IDS = [1,2]

# 父进程兜底：限制可见卡
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in GPU_IDS)

# ========== 配置区 ==========
# 模型基座路径（可根据需要修改）
BASE_MODEL_NAME = "/base/rd1/large_models/Qwen2.5-32B-Instruct"

# 定义语言和对应的模型适配器路径（全参微调目录或LoRA适配器）
#LANGUAGE_MODELS = {
#    "bodo": "/base/rd1/large_models/train_save/full/wmt_total/hy_bodo/3e-5",
#    "karbi": "/base/rd1/large_models/train_save/full/wmt_total/hy_karbi/3e-5",
#    "kokborok": "/base/rd1/large_models/train_save/full/wmt_total/hy_kokborok/3e-5",
#    "nagamese": "/base/rd1/large_models/train_save/full/wmt_total/hy_nagamese/3e-5",
#    "targin": "/base/rd1/large_models/train_save/full/wmt_total/hy_targin/3e-5",
#}

#LANGUAGE_MODELS = {
#    "bodo": "/base/rd1/large_models/train_save/full/wmt_total/hy_bodo_add/3e-5",
    #"karbi": "/base/rd1/large_models/train_save/full/wmt_total/hy_karbi_add/3e-5",
    #"kokborok": "/base/rd1/large_models/train_save/full/wmt_total/hy_kokborok_add/3e-5",
    #"nagamese": "/base/rd1/large_models/train_save/full/wmt_total/hy_nagamese_add/3e-5",
    #"targin": "/base/rd1/large_models/train_save/full/wmt_total/hy_targin_add/3e-5",
#}

LANGUAGE_MODELS = {
    #"bodo": "/base/rd1/large_models/train_save/full/dpo/dpo_bodo",
    "bodo": "/base/rd1/large_models/train_save/full/test_hybd"
    #"karbi": "/base/rd1/large_models/train_save/full/dpo/dpo_karbi",
    #"kokborok": "/base/rd1/large_models/train_save/full/dpo/dpo_kokborok",
    #"nagamese": "/base/rd1/large_models/train_save/full/dpo/dpo_nagamese",
    #"targin": "/base/rd1/large_models/train_save/full/dpo/dpo_targin",
}

# 输入文件目录
INPUT_DIR = r"/base/rd1/testset/sft_testset"
#INPUT_DIR = r"/base/rd1/testset/sft_add_testset"

# 输出目录
OUTPUT_DIR = r"/base/rd1/testset/sft_test_predictions"

# 预测参数
BATCH_SIZE = 128       # vLLM 内部连续批处理
DEFAULT_TEMPERATURE = 0.0  # do_sample=False，即temperature=0
TEMP_STEP = 0.1        # 每次增加的温度
MAX_NEW_TOKENS = 256
MAX_MODEL_LEN = 1300   # 输入 2048 + 输出 128
GPU_MEM_UTIL = 0.90    # vLLM 占用的显存比例
REPETITION_PENALTY = 1.2  # 重复惩罚
# ============================


def process_shard(shard_index, gpu_id, shard, model_path, result_queue):
    """处理一个数据分片"""
    # 每个子进程只看到一张物理卡（在 vLLM 任何 import 之前生效）
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[GPU {gpu_id}] 加载模型: {os.path.basename(model_path)} ...")

    # 检测模型路径是否为适配器（LoRA）或全参微调模型
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_lora:
        # LoRA适配器模式
        from vllm.lora.request import LoRARequest
        llm = LLM(
            model=BASE_MODEL_NAME,
            enable_lora=True,
            max_lora_rank=16,
            tensor_parallel_size=1,
            dtype="float16",
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        lora_request = LoRARequest("adapter", 1, model_path)
    else:
        # 全参微调模型模式
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="float16",
            gpu_memory_utilization=GPU_MEM_UTIL,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            enforce_eager=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        lora_request = None

    print(f"[GPU {gpu_id}] 模型加载完成")

    def build_prompt(instruction):
        """构建prompt"""
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    processed_data = []
    all_prompts = []
    all_original_items = []

    # 批量构建prompt
    for item in shard:
        if isinstance(item, dict) and "instruction" in item:
            prompt = build_prompt(item["instruction"])
        else:
            # 如果不是标准格式，尝试其他可能的字段
            for key in ["input", "question", "text"]:
                if key in item:
                    prompt = build_prompt(item[key])
                    break
            else:
                prompt = build_prompt(str(item))

        all_prompts.append(prompt)
        all_original_items.append(item)

    print(f"[GPU {gpu_id}] 开始生成预测，共 {len(shard)} 条数据")

    for batch_start in tqdm(
        range(0, len(shard), BATCH_SIZE),
        desc=f"GPU {gpu_id}",
        position=shard_index,
    ):
        batch_end = batch_start + BATCH_SIZE
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_items = all_original_items[batch_start:batch_end]

        # 第一轮预测：使用temperature=0（do_sample=False）
        sampling_params = SamplingParams(
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=MAX_NEW_TOKENS,
            repetition_penalty=REPETITION_PENALTY,
        )

        # 生成预测
        if is_lora and lora_request:
            outputs = llm.generate(
                prompts=batch_prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
                use_tqdm=False,
            )
        else:
            outputs = llm.generate(
                prompts=batch_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

        pending_indices = []
        predictions = []

        # 处理第一轮预测结果
        for idx, out in enumerate(outputs):
            resp = out.outputs[0].text.strip()
            predictions.append(resp)

            # 如果预测为空，标记为需要重试
            if not resp:
                pending_indices.append(idx)

        # 如果有空的预测，尝试增加温度重新生成
        attempt = 1
        while pending_indices:
            temperature = DEFAULT_TEMPERATURE + TEMP_STEP * attempt
            print(f"[GPU {gpu_id}] 第 {attempt} 轮重试，{len(pending_indices)} 条空预测，温度升至 {temperature}")

            retry_sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=MAX_NEW_TOKENS,
                repetition_penalty=REPETITION_PENALTY,
            )

            retry_prompts = [batch_prompts[i] for i in pending_indices]

            # 重新生成
            if is_lora and lora_request:
                retry_outputs = llm.generate(
                    prompts=retry_prompts,
                    sampling_params=retry_sampling_params,
                    lora_request=lora_request,
                    use_tqdm=False,
                )
            else:
                retry_outputs = llm.generate(
                    prompts=retry_prompts,
                    sampling_params=retry_sampling_params,
                    use_tqdm=False,
                )

            new_pending = []
            for list_idx, global_idx in enumerate(pending_indices):
                resp = retry_outputs[list_idx].outputs[0].text.strip()
                if resp:
                    predictions[global_idx] = resp
                else:
                    new_pending.append(global_idx)

            pending_indices = new_pending
            attempt += 1

            # 最多尝试5次
            if attempt > 5:
                break

        # 添加预测结果到原始数据
        for idx, (item, pred) in enumerate(zip(batch_items, predictions)):
            # 创建新字典，保留所有原始字段并添加predict字段
            if isinstance(item, dict):
                new_item = dict(item)
            else:
                new_item = {"original": str(item)}
            new_item["predict"] = pred
            processed_data.append(new_item)

    print(f"[GPU {gpu_id}] 完成，处理了 {len(processed_data)} 条数据")
    result_queue.put((shard_index, processed_data))


def split_data(data, n):
    """将数据分成n个分片"""
    size = (len(data) + n - 1) // n
    return [data[i * size: (i + 1) * size] for i in range(n)]


def process_language(language, model_path, input_file, output_file):
    """处理一种语言的数据"""
    print(f"\n{'='*60}")
    print(f"处理语言: {language}")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模型路径: {model_path}")
    print(f"{'='*60}")

    # 检查输入文件
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return False

    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"警告: 模型路径不存在: {model_path}")
        # 尝试使用默认路径格式
        default_path = f"/base/rd1/large_models/train_save/full/wmt_total/hy_{language}/3e-5"
        if os.path.exists(default_path):
            model_path = default_path
            print(f"使用默认路径: {model_path}")
        else:
            print(f"错误: 模型路径和默认路径都不存在")
            return False

    # 读取数据
    print(f"读取数据：{input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件错误: {e}")
        return False

    print(f"共 {len(data)} 条数据")

    # 如果数据量少，只用一张GPU
    if len(data) < BATCH_SIZE * 2:
        gpu_count = 1
        print(f"数据量少，只使用 GPU {GPU_IDS[0]}")
    else:
        gpu_count = min(len(GPU_IDS), max(1, len(data) // (BATCH_SIZE * 10)))
        print(f"使用 {gpu_count} 张GPU")

    used_gpus = GPU_IDS[:gpu_count]
    shards = split_data(data, gpu_count)

    for gpu_id, shard in zip(used_gpus, shards):
        print(f"  GPU {gpu_id}: {len(shard)} 条")

    # 创建多进程
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    processes = []

    for shard_index, (gpu_id, shard) in enumerate(zip(used_gpus, shards)):
        p = mp.Process(
            target=process_shard,
            args=(shard_index, gpu_id, shard, model_path, result_queue),
        )
        p.start()
        processes.append(p)

    # 收集结果
    results = {}
    while len(results) < gpu_count:
        try:
            shard_index, processed_data = result_queue.get(timeout=5)
            results[shard_index] = processed_data
        except queue.Empty:
            dead = [(i, p.exitcode) for i, p in enumerate(processes)
                    if not p.is_alive() and p.exitcode != 0]
            if dead:
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                print(f"错误: 子进程异常退出（shard_index, exitcode）：{dead}")
                return False

    for p in processes:
        p.join()

    # 合并结果
    all_processed_data = []
    for i in range(gpu_count):
        all_processed_data.extend(results.get(i, []))

    print(f"✅ 预测完成！共处理 {len(all_processed_data)} 条数据")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed_data, f, ensure_ascii=False, indent=4)
    print(f"已写入：{output_file}")

    return True


def main():
    """主函数"""
    print("开始批量预测...")
    print(f"配置参数:")
    print(f"  GPU设备: {GPU_IDS}")
    print(f"  默认温度: {DEFAULT_TEMPERATURE} (do_sample=False)")
    print(f"  温度步长: {TEMP_STEP}")
    print(f"  重复惩罚: {REPETITION_PENALTY}")
    print(f"  批量大小: {BATCH_SIZE}")
    print(f"  最大新词元: {MAX_NEW_TOKENS}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 处理所有语言文件
    input_files = list(Path(INPUT_DIR).glob("*_final_test.json"))
    print(f"\n找到 {len(input_files)} 个输入文件:")
    for f in input_files:
        print(f"  - {f.name}")

    successful = 0
    failed = 0

    for input_file in input_files:
        # 提取语言名称
        filename = input_file.name
        for language in LANGUAGE_MODELS.keys():
            if language.lower() in filename.lower():
                break
        else:
            # 尝试从文件名推断
            language = filename.split("_")[0].lower()
            print(f"注意: 从文件名推断语言为: {language}")

        # 构建输出文件路径
        output_file = os.path.join(OUTPUT_DIR, f"{language}_predicted.json")

        # 获取模型路径
        model_path = LANGUAGE_MODELS.get(language)
        if not model_path:
            print(f"警告: 未找到 {language} 的模型配置，跳过")
            failed += 1
            continue

        # 处理该语言
        if process_language(language, model_path, str(input_file), output_file):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"批量预测完成！成功: {successful}, 失败: {failed}")
    print(f"预测结果保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

