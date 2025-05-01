from trl import IterativeSFTTrainer
from trl.trainer.utils import DPODataCollatorWithPadding
from dataclasses import dataclass
import torch
import math

from transformers.trainer_utils import seed_worker
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from functools import wraps


@dataclass
class CustomSTaRConfig(TrainingArguments):
    eval_samples: int = 32
    min_sft_part: float = 0.1


class CustomSTaRTrainer(IterativeSFTTrainer):
    def __init__(self, reward_funcs=None, processing_class=None, **kwargs):
        self.reward_funcs = reward_funcs

        self.data_collator_2 = DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id)
        super().__init__(processing_class=processing_class, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch = None):
        inputs = self._filter(inputs)

        input_ids, loss_mask = get_loss_masks(inputs, self.processing_class)
        labels = [x.clone() for x in input_ids]
        for label, mask in zip(labels, loss_mask): label[mask == 0] = -100
        attention_mask = [torch.ones_like(x) for x in input_ids]
        self.step(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        return torch.tensor(0.0, device=self.model.device)
    
    def _filter(self, inputs):
        rewards = torch.tensor([c.game.score(n) for c, n in zip(inputs['conversation'], inputs['train_llm_num'])])

        sorted_rewards, sorted_indices = torch.sort(rewards, descending=True)
        i = math.ceil(len(rewards)*self.args.min_sft_part)
        while i < len(rewards) and sorted_rewards[i] == sorted_rewards[i - 1]: i += 1
        selected_indices = sorted_indices[:i]

        return {k: [v[i] for i in selected_indices] for k, v in inputs.items()}
    
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

