from trl import IterativeSFTTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
from dataclasses import dataclass
import torch
import math

from transformers.trainer_utils import seed_worker, SaveStrategy
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from functools import wraps


@dataclass
class CustomSTaRConfig(TrainingArguments):
    eval_samples: int = 32
    min_sft_part: float = 0.1
    star_batch_size: int = 1

    reward_weights: list = None

class CustomSTaRTrainer(IterativeSFTTrainer):
    def __init__(self, reward_funcs=None, processing_class=None, **kwargs):
        self.data_collator_2 = DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id)
        super().__init__(processing_class=processing_class, **kwargs)

        self.stats = {
            "SFT_runs %": [],
            "reward": [],
            "batch_size": [],
        }
        
        
        self.reward_funcs = reward_funcs
        if isinstance(reward_funcs, list):
            if self.args.reward_weights is None: self.reward_weights = [1.] * len(reward_funcs)
            assert len(reward_funcs) == len(self.args.reward_weights), "Number of reward_funcs, and reward_weights should be the same"

            for i in range(len(reward_funcs)):
                self.stats[f"rewards/{reward_funcs[i].__name__}"] = []

    def training_step(self, model, inputs, num_items_in_batch = None):
        inputs = self._filter(inputs)

        input_ids, loss_mask = get_loss_masks(inputs, self.processing_class)
        labels = [x.clone() for x in input_ids]
        for label, mask in zip(labels, loss_mask): label[mask == 0] = -100
        attention_mask = [torch.ones_like(x) for x in input_ids]
        
        bs = self.args.star_batch_size
        for i in range(math.ceil(len(input_ids)/bs)):
            inputs_batch = input_ids[i*bs:(i+1)*bs]
            attention_batch = attention_mask[i*bs:(i+1)*bs]
            labels_batch = labels[i*bs:(i+1)*bs]
            self.step(input_ids=inputs_batch, attention_mask=attention_batch, labels=labels_batch)

        return torch.tensor(0.0, device=self.model.device)
    
    def _filter(self, inputs):
        original_inputs = inputs.copy()
        inputs['completions'] = inputs['prompts'] = None
        if isinstance(self.reward_funcs, list):
            reward_per_func = [torch.Tensor(func(**inputs)) for func in self.reward_funcs]
            rewards = sum(
                    self.args.reward_weights[i] * reward_per_func[i]
                    for i in range(len(self.judge))
                )
        else:
            rewards = torch.Tensor(self.reward_funcs(**inputs))

        sorted_rewards, sorted_indices = torch.sort(rewards, descending=True)
        i = math.ceil(len(rewards)*self.args.min_sft_part)
        while i < len(rewards) and sorted_rewards[i] == sorted_rewards[i - 1]: i += 1
        selected_indices = sorted_indices[:i]

        self.stats["SFT_runs %"].append(i / len(rewards))
        self.stats["reward"].append(rewards.mean().item())
        self.stats["batch_size"].append(len(rewards))

        if isinstance(self.judge, list):
            for i, reward_func in enumerate(self.judge):
                self.stats[f"rewards/{reward_func.__name__}"].append(reward_per_func[i].mean().item())

        return {k: [v[i] for i in selected_indices] for k, v in original_inputs.items()}
    
    # copied from OnlineDPOTrainer (except for the collator)
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator_2
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    
    # copied from Trainer
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs = {}

            tr_loss_scalar = self._nested_gather(self.tr_loss).mean().item()
            # reset tr_loss to zero
            self.tr_loss -= self.tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()
            
            ######## Add our metrics ########
            for key, val in self.stats.items():
                logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats
            #################################

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)



def get_loss_masks(inputs, tokenizer):
    texts, intervals_list = [], []
    for c, n in zip(inputs['conversation'], inputs['train_llm_num']):
        train_player = c.get_player(n)
        texts.append(train_player.pov)
        intervals_list.append(train_player.get_talk_intervals())
    
    batch_input_ids, batch_loss_masks = [], []
    for text, intervals in zip(texts, intervals_list):
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids, offsets = encoding['input_ids'], encoding['offset_mapping']

        loss_mask = torch.zeros(len(offsets))
        i = j = 0  # token/interval pointer
        while i < len(offsets) and j < len(intervals):
            if offsets[i][1] <= intervals[j][0]: i += 1
            elif offsets[i][0] >= intervals[j][1]: j += 1
            else:
                loss_mask[i] = 1
                i += 1

        batch_input_ids.append(torch.tensor(input_ids))
        batch_loss_masks.append(loss_mask)

    return batch_input_ids, batch_loss_masks

