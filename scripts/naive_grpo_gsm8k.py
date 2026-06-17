import os

from numpy import dtype
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append("../")
import yaml
from typing import Literal
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tests.adapters import run_grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_config(
        config_path: str = "naive_grpo_configs.yaml"
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_gsm8k_data(
        data_path: str = Literal["../data/gsm8k/train.jsonl", "../data/gsm8k/test.jsonl"]
) -> tuple[list[str], list[str]]:
    questions, answers = [], []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["question"])
            answers.append(data["answer"])
    return questions, answers

def duplicate_data(
    data: list[str],
    repeat: int
) -> list[str]:
    return [item for item in data for _ in range(repeat)]
    

def main():
    config_path = "naive_grpo_configs.yaml"
    config = load_config(config_path)

    raw_train_questions, raw_train_answers = load_gsm8k_data(
        data_path="../data/gsm8k/train.jsonl"
    )
    raw_test_questions, raw_test_answers = load_gsm8k_data(
        data_path="../data/gsm8k/test.jsonl"
    )

    logger.debug(f"type(raw_train_questions) is {type(raw_train_questions)}")
    logger.debug(f"type(raw_train_question[0]) is {type(raw_train_questions[0])}")

    duplicated_train_questions = duplicate_data(
        data=raw_train_questions,
        repeat=config["group_size"]
    )
    duplicated_train_answers = duplicate_data(
        data=raw_train_answers,
        repeat=config["group_size"]
    )

    prompt_type = config["prompt_type"]
    server = VLLMServer(
        model_id=config["model"], 
        gpu=5, 
        seed=0,
        gpu_memory_utilization=0.9
    )
    server.start()

    model = AutoModelForCausalLM.from_pretrained(
        config["model"], 
        dtype=torch.bfloat16
    ).to(config["device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    n_train_examples = min(len(duplicated_train_questions), config["n_train_examples"])

    for i in range(0, n_train_examples, config["train_batch_size"]):

        rollout_responses: list[VLLMCompletion] = server.generate_completions(
            prompts=duplicated_train_questions[i: i+config["train_batch_size"]],
            sampling_params={
                "temperature": config["sampling_temperature"],
                "top_p": 1.0,
                "max_tokens": config["sampling_max_tokens"],
                "n": 1,
                "seed": 0,
                "stop": ["</answer>"] if prompt_type == "r1_zero" else None,
                "include_stop_str_in_output": True if prompt_type == "r1_zero" else False
            }
        )
        logger.debug(f"type(rollout_response) is {type(rollout_responses)}")
        logger.debug(f"type(rollout_response[0]) is {type(rollout_responses[0])}")

        rollout_responses_texts = [item.text for item in rollout_responses]


        loss, metadata = run_grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.0),
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            max_grad_norm=config["max_grad_norm"],
            reward_fn=question_only_reward_fn if prompt_type == "r1_zero_three_shot" else r1_zero_reward_fn,
            repeated_prompts=duplicated_train_questions[i:i+config["train_batch_size"]],
            rollout_responses=rollout_responses_texts,
            repeated_ground_truths=duplicated_train_answers[i:i+config["train_batch_size"]],
            group_size=config["group_size"],
            baseline=config["baseline"],
            advantage_eps=config["advantage_eps"],
            advantage_normalizer=config["advantage_normalizer"],
            importance_reweighting_method=config["importance_reweighting_method"],
            loss_normalization="sequence",
            device=config["device"]
        )

        print(f"loss: {loss}")
        print(f"metadata: \n{(metadata)}")

    server.stop()

if __name__ == "__main__":
    main()