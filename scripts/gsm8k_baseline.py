'''
1. 加载模型和 tokenizer（用 HuggingFace）
2. 加载 GSM8K 测试集（data/gsm8k/test.jsonl）
3. 对每个 prompt 类型：
   - 读取对应的 prompt 模板文件
   - 对每个测试问题，将问题填入模板，得到完整的 prompt 字符串
   - 用 vLLM 生成回答（temperature=1.0, top_p=1.0, max_tokens=512）
   - 对于 r1_zero 类 prompt，设置 stop = ["</answer>"]，并 include_stop_str_in_output=True
   - 用相应的 reward_fn 打分（r1_zero_reward_fn 或 question_only_reward_fn）
   - 统计三类结果：
        (1) format_reward=1 且 answer_reward=1 → 正确且格式正确
        (2) format_reward=1 且 answer_reward=0 → 格式正确但答案错
        (3) format_reward=0 且 answer_reward=0 → 格式错（答案自然也算错）
4. 输出统计表，并保存一些示例（prompt + 模型原始输出 + 解析结果）
'''
import sys
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
import logging
from tqdm import tqdm
import json
sys.path.append("../")
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_gsm8k_test_set(
        file_path: str
    ) -> list[dict]:
    test_data = []
    with open(file_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data

def process_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)

def extract_ground_truths(answers: list[str]) -> list[str]:
    ground_truths = []
    for answer in answers:
        ground_truth_begin_idx = answer.rfind("####")  # 假设答案格式为 "#### 正确答案"
        if ground_truth_begin_idx != -1:
            ground_truths.append(answer[ground_truth_begin_idx + 4:].strip())
        else:
            warning_msg = f"Warning: Answer format unexpected, cannot extract ground truth. Answer: {answer}"
            logger.warning(warning_msg)
            ground_truths.append("")  # 如果格式不对，暂时放一个空字符串
    return ground_truths

def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K Baseline Evaluation")
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-0425-1B", help="模型名称或路径")
    parser.add_argument("--test_file", type=str, default="../data/gsm8k/test.jsonl", help="GSM8K 测试集文件路径")
    parser.add_argument("--prompt_dir", type=str, default="../cs336_alignment/prompts/", help="Prompt 模板文件夹路径")
    parser.add_argument("--output_dir", type=str, default="../outputs/gsm8k_baseline/", help="输出结果保存文件夹路径")
    return parser.parse_args()

def main():
    args = parse_args()
    server = VLLMServer(
        model_id=args.model_name,
        gpu=4,
        gpu_memory_utilization=0.5
    )
    server.start()

    test_data = load_gsm8k_test_set(args.test_file)
    
    prompt_templates_path = {
        "r1_zero": os.path.join(args.prompt_dir, "r1_zero.prompt"),
        "question_only": os.path.join(args.prompt_dir, "question_only.prompt"),
        "r1_zero_three_shot": os.path.join(args.prompt_dir, "r1_zero_three_shot_gsm8k.prompt")
    }

    results = { "r1_zero": {
                    "correct_format_and_answer": 0, 
                    "correct_format_wrong_answer": 0, 
                    "wrong_format": 0,
                    "examples": []
                },
                "question_only": {
                    "correct_format_and_answer": 0, 
                    "correct_format_wrong_answer": 0, 
                    "wrong_format": 0,
                    "examples": []
                },
                "r1_zero_three_shot": {
                    "correct_format_and_answer": 0, 
                    "correct_format_wrong_answer": 0, 
                    "wrong_format": 0,
                    "examples": []
                }
            }

    for prompt_type, template_path in prompt_templates_path.items():
        with open(template_path, 'r') as f:
            template = f.read()
        
        prompts = []
        answers = []
        for item in test_data:
            question = item['question']
            prompt = process_prompt(template, question)
            prompts.append(prompt)

            answer = item['answer']
            answers.append(answer)

        ground_truths = extract_ground_truths(answers)

        sample_params = {
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "n": 1,
            "seed": 0,
            "stop": ["</answer>"] if prompt_type == "r1_zero" else None,
            "include_stop_str_in_output": True if prompt_type == "r1_zero" else False
        }

        completions = server.generate_completions(
            prompts=prompts,
            sampling_params=sample_params,
            batch_size=8
        )

        if prompt_type == "r1_zero":
            reward_fn = r1_zero_reward_fn
        else:            
            reward_fn = question_only_reward_fn
        
        for completion, ground_truth in zip(completions, ground_truths):
            result = reward_fn(completion.text, ground_truth)

            if result["reward"] == 1.0:
                results[prompt_type]["correct_format_and_answer"] += 1
            elif result["format_reward"] == 1.0:
                results[prompt_type]["correct_format_wrong_answer"] += 1
            else:
                results[prompt_type]["wrong_format"] += 1
            results[prompt_type]["examples"].append(
                json.dumps({ "completion": completion.text, "ground_truth": ground_truth, "result": result }, 
                ensure_ascii=False, 
                indent=2)
            )

    print("Evaluation Results:")
    for prompt_type, res in results.items():
        print(f"Prompt Type: {prompt_type}")
        print(f"  Correct Format & Answer: {res['correct_format_and_answer']}")
        print(f"  Correct Format & Wrong Answer: {res['correct_format_wrong_answer']}")
        print(f"  Wrong Format: {res['wrong_format']}")
        print(f"  Total: {res['correct_format_and_answer'] + res['correct_format_wrong_answer'] + res['wrong_format']}")
        print(f"  Examples: {res['examples'][:5]}")  # 打印前5个示例

    server.stop()

if __name__ == "__main__":
    main()