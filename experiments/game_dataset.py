from transformers import TrainerCallback
from torch.utils.data import Dataset
import torch

from copy import deepcopy
from math import floor
import random

from conversation_manager import ConversationManager, autoassign
from game_utils import masked_call
from games import get_game

@torch.no_grad()
def finish_conversations(conversations, train_llm, opponent_llm, train_llm_num, config):
    gen_conf_trained = {
        "max_new_tokens" : config.train.max_completion_length,
        "temperature" : config.train.trained_temperature,
        "do_sample" : config.train.trained_temperature > 0
    }
    gen_conf_opponent = {
        "max_new_tokens" : config.train.max_completion_length,
        "top_p" : config.train.opponent_top_p,
        "temperature" : config.train.opponent_temperature,
        "do_sample" : config.train.opponent_temperature > 0
    }

    for interaction_idx in range(1, 1+2*config.dataset.max_interactions):
        if all(c.finished() for c in conversations):
            break

        curr_llm = train_llm if interaction_idx%2 == train_llm_num else opponent_llm
        curr_gen_conf = gen_conf_trained if interaction_idx%2 == train_llm_num else gen_conf_opponent
        actions = masked_call(
                    lambda x: curr_llm.generate(x, **curr_gen_conf),
                    [c.get_query() for c in conversations],
                    [c.num_interactions == interaction_idx-1 and not c.finished() for c in conversations]
                )
        for idx, action in enumerate(actions):
            if action:
                conversations[idx].turn(action)
    
    return conversations


class GameDataset(Dataset):
    @autoassign
    def __init__(self, train_llm, opponent_llm, config):
        with open(config.prompts.folder + config.prompts.initial, "r") as file:
            self.initial_prompt = file.read()
        with open(config.prompts.folder + config.prompts.other_moved, "r") as file:
            self.other_moved_prompt = file.read()
        self.Game = get_game(config.dataset.game_name)
        self.create_batch()

    def __len__(self):
        return self.config.dataset.samples_per_epoch

    # generate root conversation and all sub-conversations
    def __getitem__(self, idx):
        if idx >= len(self.batch):
            print("idx out of range, repeating sample")
            idx %= len(self.batch)
        conv = self.batch[idx]
        return {"prompt": conv.get_query(), "conversation": conv, "train_llm_num" : self.train_llm_num}

    def create_batch(self):
        print("\nCreating batch", flush=True)

        conversations = [ConversationManager(
            self.initial_prompt, self.other_moved_prompt,
            self.config.dataset.player_1_name, self.config.dataset.player_2_name,
            self.Game, self.config.dataset.max_interactions
        ) for _ in range(self.config.dataset.num_root_generations) ]
        self.train_llm_num = 1 if self.config.dataset.trained_player == 1 else 0 if self.config.dataset.trained_player == 2 else round(random.random())
        conversations = finish_conversations(conversations, self.train_llm, self.opponent_llm, self.train_llm_num, self.config)

        all_subconversations = [list(c.get_subconversations(self.train_llm_num)) for c in conversations]
        self.batch = sum(all_subconversations, [])
        self.full_conversations = conversations
        random.shuffle(self.batch)

        print("Batch created", flush=True)


class OutdateDatasetCallback(TrainerCallback):
    def __init__(self, gameDataset):
        super().__init__()
        self.gameDataset = gameDataset

    def on_epoch_end(self, args, state, control, **kwargs):
        self.gameDataset.create_batch()

class MetricsLogger(TrainerCallback):
    @autoassign
    def __init__(self, gameDataset, metics_file, conversation_file, config):
        print(f"Configuration:\n{config}\n\n\n", file=self.metics_file, flush=True)
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        print("EPOCH", floor(state.epoch), "- STEP", state.global_step, file=self.metics_file, flush=True)
        print(logs, file=self.metics_file)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        print("EPOCH", state.epoch, file=self.conversation_file)
        for c in self.gameDataset.full_conversations:
            print(c.full_conversation, '\n\n', file=self.conversation_file)

def reward(prompts, completions, conversation, train_llm_num, Game, train_llm, opponent_llm, config):
    print("\nComputing rewards", flush=True)
    conversations = [deepcopy(c) for c in conversation]
    for idx, action in enumerate(completions): conversations[idx].turn(action)

    conversation = finish_conversations(conversations, train_llm, opponent_llm, train_llm_num, config)

    moves = [c.get_moves() for c in conversation]
    moves = [w if n == 1 else (w[1], w[0]) for w, n in zip(moves, train_llm_num)]
    rewards = [Game.score(w[0], w[1]) for w in moves]

    print("Rewards computed", flush=True)
    return rewards
