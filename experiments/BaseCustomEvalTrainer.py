from transformers.trainer_utils import EvalLoopOutput
from transformers import Trainer
from trl import GRPOConfig

from collections import defaultdict
from dataclasses import dataclass
import torch
import math

from metrics import wordBasedLoss

@dataclass
class CustomGRPOConfig(GRPOConfig):
    eval_samples: int = 32


# So far I only care about dataloader
# "internal_state_loss" is still to be tracked

class BaseCustomEvalTrainer:
    def __init__(self, *args, **kwargs):
        assert Trainer in self.__class__.mro(), "BaseCustomEvalTrainer is meant to be inherited with a subclass of trl.Trainer\nThat is: class YourTrainer(BaseCustomEvalTrainer, BaseTrainer): pass, where BaseTrainer is a subclass of Trainer, like GRPOTrainer"
        super().__init__(*args, **kwargs)

        self.root_gens_iteration = self.args.eval_samples
        self.num_oom = 1

    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
    
        dataset = dataloader.dataset
        Game = dataset.Game
        num_iterations = math.floor(self.args.eval_samples/self.root_gens_iteration)

        assert description == "Evaluation", "Oops, evaluation description is not Evaluation"

        print("\n\nEvaluation starts", flush=True)
        metrics = defaultdict(float)
        try:
            for i in range(num_iterations):
                _, full_conversations, train_llm_num = dataset.create_eval_batch(num_root_generations = self.root_gens_iteration)
                conversations_text = [c.full_conversation for c in full_conversations]
                games = [c.game for c in full_conversations]

                metrics["reward"] += sum(g.score(train_llm_num) for g in games)
                metrics["opponent_reward"] += sum(g.score(3-train_llm_num) for g in games)
                metrics["num_interactions"] += sum(c.num_interactions for c in full_conversations)
                metrics["conv_length (tokens)"] += sum(len(tokens) for tokens in self.processing_class(conversations_text)['input_ids'])
                metrics["finished_by_error"] += sum(1 for g in games if g.is_error())

                game_metrics = Game.game_metrics(games, train_llm_num)
                for k, v in game_metrics.items():
                    metrics[k] += v

                metrics["word_based_loss"] += sum(wordBasedLoss(c, train_llm_num) for c in full_conversations)

            num_samples = self.root_gens_iteration*num_iterations
            for k in metrics:
                metrics[k] /= num_samples
            
            metrics["normalized_relative_advantage"] = 0. if metrics["reward"] == metrics["opponent_reward"] == 0 else \
                (metrics["reward"] - metrics["opponent_reward"]) / (metrics["reward"] + metrics["opponent_reward"])

            # Prefix all keys with metric_key_prefix + '_'
            metric_names = list(metrics.keys())
            for key in metric_names:
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            return EvalLoopOutput(metrics=metrics, predictions = None, label_ids=None, num_samples=num_samples)
        
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM error during evaluation, # {self.num_oom}")
            print(f"Reducing root_gens_iteration from {self.root_gens_iteration} to", end=" ")

            self.num_oom += 1
            self.root_gens_iteration = min(math.floor(self.args.eval_samples/self.num_oom), self.root_gens_iteration-1)
            
            print(self.root_gens_iteration)

            if self.root_gens_iteration <= 0:
                print("EXIT: Eval samples is 0 (number of root generations per iteration is 0)")
                raise AssertionError("Not enough memory to evaluate")
            
            return self.evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
