from ast import Not
import re
import argparse
from matplotlib import pyplot as plt
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--log_path", type=str, default="../naive_grpo_gsm8k.log", help="Path to the log directory")
    parser.add_argument("--output_path", type=str, default="../plots", help="Path to save the plots")
    return parser.parse_args()

def extract_metrics(
    log_path: str
) -> dict[str, dict[int, dict[str, float]]]:
    train_metrics = {}
    val_metrics = {}
    train_loss_reward_record: dict[str, float|None] = {"train_loss": None, "train_format_reward": None, "train_total_reward": None}
    train_step = 0
    with open(log_path, "r") as f:
        for line in f:
            match = re.search(r"Step (\d+): test_format_reward = ([\d.]+), test_answer_reward = ([\d.]+)", line)
            if match:
                step = int(match.group(1))
                test_format_reward = float(match.group(2))
                test_answer_reward = float(match.group(3))
                val_metrics[step] = {"test_format_reward": test_format_reward, "test_answer_reward": test_answer_reward}

            train_loss_match = re.search(r"^loss: (-?[\d.]+)", line)
            train_format_reward_match = re.search(r"^format_reward: ([\d.]+)", line)
            train_total_reward_match = re.search(r"^total_reward: ([\d.]+)", line)

            if train_loss_match:
                assert any(value is None for value in train_loss_reward_record.values()), "Train loss should be recorded after at least one metric is recorded."
                train_loss_reward_record["train_loss"] = float(train_loss_match.group(1))
            if train_format_reward_match:
                assert (train_loss_reward_record["train_loss"] is not None) or (train_loss_reward_record["train_total_reward"] is None), "Train format reward should be recorded after train loss."
                train_loss_reward_record["train_format_reward"] = float(train_format_reward_match.group(1))
            if train_total_reward_match:
                assert (train_loss_reward_record["train_loss"] is not None) or (train_loss_reward_record["train_format_reward"] is not None), "Train total reward should be recorded after train format reward."
                train_loss_reward_record["train_total_reward"] = float(train_total_reward_match.group(1))

            if all(value is not None for value in train_loss_reward_record.values()):
                train_step += 1
                train_metrics[train_step] = {
                    "train_loss": train_loss_reward_record["train_loss"],
                    "train_format_reward": train_loss_reward_record["train_format_reward"],
                    "train_total_reward": train_loss_reward_record["train_total_reward"]
                }
                train_loss_reward_record = {"train_loss": None, "train_format_reward": None, "train_total_reward": None}

    return {"train": train_metrics, "val": val_metrics}

def plot_metrics(
    metrics: dict[str, dict[int, dict[str, float]]], 
    output_path: str
) -> None:
    steps = list(metrics["val"].keys())
    test_format_rewards = [metrics["val"][step]["test_format_reward"] for step in steps]
    test_answer_rewards = [metrics["val"][step]["test_answer_reward"] for step in steps]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, test_format_rewards, label="Test Format Reward")
    plt.plot(steps, test_answer_rewards, label="Test Answer Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/test_metrics.png")
    plt.close()

    steps = list(metrics["train"].keys())
    train_losses = [metrics["train"][step]["train_loss"] for step in steps]
    train_format_rewards = [metrics["train"][step]["train_format_reward"] for step in steps]
    train_total_rewards = [metrics["train"][step]["train_total_reward"] for step in steps]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, train_format_rewards, label="Train Format Reward")
    plt.plot(steps, train_total_rewards, label="Train Total Reward")
    plt.xlabel("Steps")
    plt.ylabel("Metrics")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/train_metrics.png")
    plt.close()

def main():
    args = parse_args()
    metrics = extract_metrics(args.log_path)
    plot_metrics(metrics, args.output_path)

if __name__ == "__main__":
    args = parse_args()
    main()