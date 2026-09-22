import os

from traitlets import Int
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
sys.path.append("../")
import yaml
from typing import Callable, Iterator, Literal, Generator
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from cs336_alignment.core.grpo_train_step import grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer
import logging
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
from cs336_alignment.utils import get_current_time

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

writer = SummaryWriter(log_dir=f"../outputs/rft_gsm8k/{get_current_time()}", flush_secs=15)

def check_hyperparams(
    train_batch_size: int,
    group_size: int,
    n_train_examples: Int
) -> None:

    assert train_batch_size % group_size == 0, (
        f"train_batch_size ({train_batch_size}) % group_size ({group_size}) "
        f"must equal 0, but got remainder {train_batch_size % group_size}"
    )

    assert n_train_examples % (train_batch_size // group_size) == 0, (
        f"n_train_examples ({n_train_examples}) % "
        f"(train_batch_size // group_size) ({train_batch_size // group_size}) "
        f"must equal 0, but got remainder "
        f"{n_train_examples % (train_batch_size // group_size)}"
    )


def load_config(
    config_path: str = "naive_grpo_configs.yaml"
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def process_prompt(template: str, question: str) -> str:
    return template.replace("{question}", question)

def load_gsm8k_template_data(
    template: str,
    data_path: str = Literal["../data/gsm8k/train.jsonl", 
                            "../data/gsm8k/test.jsonl"],
    shuffle: bool = False,
    limit: int|None = None
) -> tuple[list[str], list[str]]:
    questions, answers = [], []
    with open(data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            templated_data = process_prompt(template=template, question=data["question"])
            questions.append(templated_data)
            answer = data["answer"].rsplit("####", 1)[-1].strip() if "####" in data["answer"] else data["answer"]
            answers.append(answer)

    if shuffle:
        combined = list(zip(questions, answers))
        random.seed(0)
        random.shuffle(combined)
        questions, answers = zip(*combined)

    if limit is not None:
        questions = questions[:limit]
        answers = answers[:limit]

    return questions, answers

def duplicate_data(
    data: list[str],
    repeat: int
) -> list[str]:
    return [item for item in data for _ in range(repeat)]


def inference_with_vllm(
    server: VLLMServer,
    prompts: list[str],
    config: dict,
    rollout_n: int = 1
) -> list[VLLMCompletion]:
    return server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": config["sampling_temperature"],
            "top_p": config["top_p"],
            "max_tokens": config["sampling_max_tokens"],
            "n": rollout_n,
            "seed": 0,
            "stop": ["</answer>"] if config["prompt_type"] == "r1_zero" or config["prompt_type"] == "r1_zero_three_shot" else None,
            "include_stop_str_in_output": True if config["prompt_type"] == "r1_zero" or config["prompt_type"] == "r1_zero_three_shot" else False
        }
    )

class dataloader:
    def __init__(
        self,
        prompts: list[str],
        batch_size: int
    ):
        self.prompts = prompts
        self.batch_size = batch_size
        self._batch_generator = self._generate_batches()
    
    def _generate_batches(self) -> Iterator[list[str]]:
        for i in range(0, len(self.prompts), self.batch_size):
            yield self.prompts[i:i + self.batch_size]
    
    def __call__(self) -> Generator[list[str], None, None]:
        try:
            return next(self._batch_generator)
        except StopIteration:
            return None

def run_test(
    server: VLLMServer,
    questions: list[str],
    answers: list[str],
    config: dict,
    reward_fn: Callable[[list[str], list[str]], dict[str, float]]
) -> tuple[float, float]:
    
    responses = inference_with_vllm(
        server=server,
        prompts=questions,
        config=config,
        rollout_n=1
    )

    format_reward, answer_reward = 0.0, 0.0
    responses_texts = [item.text for item in responses]
    for response, answer in zip(responses_texts, answers):
        result_dict: dict[str, float] = reward_fn(response, answer)
        format_reward += result_dict["format_reward"]
        answer_reward += result_dict["answer_reward"]

    return format_reward / len(questions), answer_reward / len(questions)

def main():
    config_path = "rft_configs.yaml"
    config = load_config(config_path)

    check_hyperparams(config["train_batch_size"], config["group_size"], config["n_train_examples"])

    template_path = "../cs336_alignment/prompts/r1_zero.prompt"
    with open(template_path, "r") as f:
        template = f.read()

    logger.debug(f"template is {template}")

    raw_train_questions, raw_train_answers = load_gsm8k_template_data(
        template=template,
        data_path="../data/gsm8k/train.jsonl"
    )
    raw_test_questions, raw_test_answers = load_gsm8k_template_data(
        template=template,
        data_path="../data/gsm8k/test.jsonl",
        limit=config["n_val_examples"]
    )

    logger.debug(f"type(raw_train_questions) is {type(raw_train_questions)}")
    logger.debug(f"type(raw_train_question[0]) is {type(raw_train_questions[0])}")

    # 复制prompt和ground truth答案以匹配group_size
    duplicated_train_questions = duplicate_data(
        data=raw_train_questions,
        repeat=config["group_size"]
    )
    duplicated_train_answers = duplicate_data(
        data=raw_train_answers,
        repeat=config["group_size"]
    )

    server = VLLMServer(
        model_id=config["model"], 
        gpu=config["infer_device"], 
        port=8005,
        seed=0,
        gpu_memory_utilization=config["gpu_memory_utilization"]
    )
    server.start()

    logger.info("Server is successfully started!")

    model = AutoModelForCausalLM.from_pretrained(
        config["model"], 
        dtype=torch.bfloat16
    ).to(config["train_device"])
    server.init_weight_sync(policy_device=config["train_device"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.95), weight_decay=0.0)

    n_train_examples = min(len(raw_train_questions), config["n_train_examples"])
    check_hyperparams(config["train_batch_size"], config["group_size"], n_train_examples)
    train_dataloader = dataloader(
        prompts=raw_train_questions,
        batch_size=config["train_batch_size"] // config["group_size"]
    )
    tqdm_bar = tqdm(range(0, n_train_examples * config["group_size"], config["train_batch_size"]), desc="Training Progress", unit="batch")
    step = 0

    for i in tqdm_bar:
        # 推理
        
        ''' 
        Note that rollout_batch_size and train_batch_size count responses, not prompts. 
        So rollout_batch_size = train_batch_size = 256 means 32 prompts with 8 rollouts each.
        '''

        ## 同步模型
        server.sync_policy_weights(model)
        rollout_responses: list[VLLMCompletion] = inference_with_vllm(
            server=server,
            prompts=train_dataloader(),
            config=config,
            rollout_n=config["group_size"]
        )

        rollout_responses_texts = [item.text for item in rollout_responses]

        ## 训练
        train_loss, train_metadata = grpo_train_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            max_grad_norm=config["max_grad_norm"],
            reward_fn=r1_zero_reward_fn if config["prompt_type"] == "r1_zero_three_shot" or config["prompt_type"] == "r1_zero" else question_only_reward_fn,
            repeated_prompts=duplicated_train_questions[i:i+config["train_batch_size"]],
            rollout_responses=rollout_responses_texts,
            repeated_ground_truths=duplicated_train_answers[i:i+config["train_batch_size"]],
            group_size=config["group_size"],
            baseline="none",
            advantage_eps=config["advantage_eps"],
            advantage_normalizer="none",
            importance_reweighting_method=config["importance_reweighting_method"],
            loss_normalization="constant",
            normalization_constant=config["train_batch_size"] * max(len(x) for x in rollout_responses_texts),
            device=config["train_device"]
        )

        step += 1

        ## 评估
        if step % config["eval_interval"] == 0:

            # 同步模型
            server.sync_policy_weights(model)

            test_format_reward, test_answer_reward = run_test(
                server=server,
                questions=raw_test_questions,
                answers=raw_test_answers,
                config=config,
                reward_fn=r1_zero_reward_fn if config["prompt_type"] == "r1_zero_three_shot" or config["prompt_type"] == "r1_zero" else question_only_reward_fn
            )

            writer.add_scalar("[test] format_reward", test_format_reward, step)
            writer.add_scalar("[test] answer_reward", test_answer_reward, step)
            print(f"Step {step}: test_format_reward = {test_format_reward}, test_answer_reward = {test_answer_reward}")

        writer.add_scalar("[train] loss", train_loss, step)
        writer.add_scalar("[train] format_reward", train_metadata["format_reward"], step)
        writer.add_scalar("[train] total_reward", train_metadata["total_reward"], step)
        print(f"Step {step} Summary:", end=" ")
        print(f"loss: {train_loss}", end=" ")
        print(f"format_reward: {(train_metadata["format_reward"])}", end=" ")
        print(f"total_reward: {(train_metadata["total_reward"])}")

    server.stop()
    writer.close()

if __name__ == "__main__":
    main()