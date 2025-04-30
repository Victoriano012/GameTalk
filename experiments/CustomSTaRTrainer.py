from trl import SFTTrainer, SFTConfig
from dataclasses import dataclass
import torch
import math

@dataclass
class CustomSTaRConfig(SFTConfig):
    eval_samples: int = 32
    min_sft_part: float = 0.1


class CustomSTaRTrainer(SFTTrainer):
    def __init__(self, *args, reward_funcs=None, **kwargs):
        self.reward_funcs = reward_funcs
        super().__init__(*args, **kwargs)
        self.train_dataset.keep_partial_conversations = False

    def training_step(self, model, inputs, num_items_in_batch = None):
        inputs = self._filter(inputs)
        input_ids, attention_mask = get_attention_masks(inputs, self.processing_class)
        self.step(input_ids=input_ids, attention_mask=attention_mask)
    
    def _filter(self, inputs):
        rewards = torch.tensor([c.game.score(n) for c, n in zip(inputs['conversation'], inputs['train_llm_num'])])

        sorted_rewards, sorted_indices = torch.sort(rewards, descending=True)
        i = math.ceil(len(rewards)*self.args.min_sft_part)
        while i < len(rewards) and sorted_rewards[i] == sorted_rewards[i - 1]: i += 1
        selected_indices = sorted_indices[:i]

        return {k: v[selected_indices] for k, v in inputs.items()}



def get_attention_masks(inputs, tokenizer):
    texts, intervals_list = [], []
    for c, n in zip(inputs['conversation'], inputs['train_llm_num']):
        train_player = c.get_player(n)
        texts.append(train_player.pov)
        intervals_list.append(train_player.get_talk_intervals())
    
    batch_input_ids, batch_attention_masks = [], []
    for text, intervals in zip(texts, intervals_list):
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        input_ids, offsets = encoding['input_ids'], encoding['offset_mapping']
        
        attention_mask = [0] * len(offsets)
        i = j = 0  # token/interval pointer
        while i < len(offsets) and j < len(intervals):
            if offsets[i][1] <= intervals[j][0]: i += 1
            elif offsets[i][0] >= intervals[j][1]: j += 1
            else:
                attention_mask[i] = 1
                i += 1

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)

    return batch_input_ids, batch_attention_masks

