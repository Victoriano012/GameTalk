from transformers.trainer_utils import EvalLoopOutput
from trl import GRPOTrainer

from collections import defaultdict
import torch
import math

from metrics import wordBasedLoss


# So far I only care about dataloader
# "internal_state_loss" is still to be tracked

class TrainerCustomEval(GRPOTrainer):
    def __init__(self, eval_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_samples = eval_samples
        self.num_oom = 1

    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
    
        dataset = dataloader.dataset
        Game = dataset.Game
        eval_root_generations = math.floor(self.eval_samples/self.num_oom)

        assert description == "Evaluation", "Oops, evaluation description is not Evaluation"

        print("\n\nEvaluation starts", flush=True)
        metrics = defaultdict(float)
        try:
            for i in range(self.num_oom):
                _, full_conversations, train_llm_num = dataset.create_eval_batch(num_root_generations = eval_root_generations)
                conversations_text = [c.full_conversation for c in full_conversations]
                games = [c.game for c in full_conversations]

                metrics["reward"] += sum(g.score(train_llm_num) for g in games)
                metrics["opponent_reward"] += sum(g.score(3-train_llm_num) for g in games)
                metrics["num_interactions"] += sum(c.num_interactions for c in full_conversations)
                metrics["conv_length (tokens)"] += sum(len(tokens) for tokens in self.processing_class(conversations_text)['input_ids'])

                game_metrics = Game.game_metrics(games, train_llm_num)
                for k, v in game_metrics.items():
                    metrics[k] += v

                metrics["word_based_loss"] += sum(wordBasedLoss(c, train_llm_num) for c in full_conversations)

            num_samples = eval_root_generations*self.num_oom
            for k in metrics:
                metrics[k] /= num_samples
            metrics["normalized_relative_advantage"] = (metrics["reward"] - metrics["opponent_reward"]) / (metrics["reward"] + metrics["opponent_reward"])

            # Prefix all keys with metric_key_prefix + '_'
            for key in metrics:
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            return EvalLoopOutput(metrics=metrics, predictions = None, label_ids=None, num_samples=num_samples)
        
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM error during evaluation, # {self.num_oom}")
            print(f"Reducing eval_root_generations from {math.floor(self.eval_samples/self.num_oom)} to {math.floor(self.eval_samples/(self.num_oom+1))}")

            self.num_oom += 1
            return self.evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
