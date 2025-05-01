from transformers import TrainerCallback
from torch.utils.data import Dataset
import torch

from copy import deepcopy
from math import floor
import random

from conversation_manager import ConversationManager
from utils import masked_call
from games import get_game
from utils import autoassign

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

        curr_llm = train_llm if interaction_idx%2 == train_llm_num%2 else opponent_llm
        curr_gen_conf = gen_conf_trained if interaction_idx%2 == train_llm_num%2 else gen_conf_opponent
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
    def __init__(self, train_llm, opponent_llm, data, config, keep_partial_conversations=True):
        with open(config.prompts.folder + config.prompts.initial_A, "r") as file:
            self.initial_prompt_A = file.read()
        with open(config.prompts.folder + config.prompts.initial_B, "r") as file:
            self.initial_prompt_B = file.read()
        with open(config.prompts.folder + config.prompts.intermediate, "r") as file:
            self.intermediate_prompt = file.read()
        self.Game = get_game(config.dataset.game_name)
        self.update_batch()
        self.eval_batch_shown = True
        self.column_names = ["prompt", "conversation", "train_llm_num"]

    def __len__(self):
        if not self.keep_partial_conversations:
            return self.config.dataset.num_root_generations
        return self.config.dataset.samples_per_epoch

    # generate root conversation and all sub-conversations
    def __getitem__(self, idx):
        if idx >= len(self.batch):
            print("idx out of range, repeating sample")
            idx %= len(self.batch)
        conv = self.batch[idx]
        return {"prompt": conv.get_query(), "conversation": conv, "train_llm_num" : self.train_llm_num}

    def update_batch(self):
        self.batch, self.full_conversations, self.train_llm_num = self._create_batch()
        if not self.keep_partial_conversations:
            self.batch = self.full_conversations
        random.shuffle(self.batch)

    def create_eval_batch(self, num_root_generations=None):
        output = self._create_batch(num_root_generations=num_root_generations)
        _, self.eval_batch, _ = output
        self.eval_batch_shown = False
        return output

    def _create_batch(self, num_root_generations=None):
        print("\nCreating batch", flush=True)

        if num_root_generations is None:
            num_root_generations = self.config.dataset.num_root_generations
        
        if self.data:
            length = len(next(iter(self.data.values())))
            if num_root_generations <= length:
                indices = random.sample(range(length), k=num_root_generations)
            else:
                indices = random.choices(range(length), k=num_root_generations)
            initial_kwargs = [{key: values[i] for key, values in self.data.items()} for i in indices]
        else:
            initial_kwargs = [{} for _ in range(num_root_generations)]

        train_llm_num = self.config.dataset.trained_player
        if not isinstance(train_llm_num, int):
            train_llm_num = round(random.random())+1
        player_A_num = self.config.dataset.player_A_num
        if not isinstance(player_A_num, int):
            player_A_num = round(random.random())+1
        for kwargs in initial_kwargs:
            kwargs["player_A_num"] = player_A_num
        initial_prompt_1 = self.initial_prompt_A if kwargs["player_A_num"] == 1 else self.initial_prompt_B
        initial_prompt_2 = self.initial_prompt_A if kwargs["player_A_num"] == 2 else self.initial_prompt_B

        conversations = [ConversationManager(
            initial_prompt_1, initial_prompt_2, self.intermediate_prompt,
            self.config.dataset.player_1_name, self.config.dataset.player_2_name,
            self.Game, max_interact = self.config.dataset.max_interactions,
            **initial_kwargs[i]
        ) for i in range(num_root_generations) ]

        conversations = finish_conversations(conversations, self.train_llm, self.opponent_llm, train_llm_num, self.config)

        all_subconversations = [list(c.get_subconversations(train_llm_num)) for c in conversations]
        print("Batch created", flush=True)

        return sum(all_subconversations, []), conversations, train_llm_num


class OutdateDatasetCallback(TrainerCallback):
    def __init__(self, gameDataset):
        super().__init__()
        self.gameDataset = gameDataset

    def on_epoch_end(self, args, state, control, **kwargs):
        self.gameDataset.update_batch()

class MetricsLogger(TrainerCallback):
    @autoassign
    def __init__(self, gameDataset, metics_file, conversation_file, eval_conversation_file, config):
        print(f"Configuration:\n{config}\n\n\n", file=self.metics_file, flush=True)
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        print("EPOCH", floor(state.epoch), "- STEP", state.global_step, file=self.metics_file, flush=True)
        print(logs, file=self.metics_file, flush=True)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        print("EPOCH", state.epoch, file=self.conversation_file)
        print("dataset conversations", file=self.conversation_file)
        for c in self.gameDataset.full_conversations:
            print(c.full_conversation, '\n', file=self.conversation_file)
            print("REWARD (Player-1):", c.game.score(1), file=self.conversation_file, flush=True)
            print("REWARD (Player-2):", c.game.score(2), '\n\n', file=self.conversation_file, flush=True)
        
        if not self.gameDataset.eval_batch_shown:
            print("eval conversations", file=self.conversation_file)
            for c in self.gameDataset.eval_batch:
                print(c.full_conversation, '\n', file=self.conversation_file)
                print("REWARD (Player-1):", c.game.score(1), file=self.conversation_file, flush=True)
                print("REWARD (Player-2):", c.game.score(2), '\n\n', file=self.conversation_file, flush=True)
                
            print("EPOCH", state.epoch, file=self.eval_conversation_file)
            for c in self.gameDataset.eval_batch:
                print(c, '\n', file=self.eval_conversation_file)
                print("REWARD (Player-1):", c.game.score(1), file=self.eval_conversation_file, flush=True)
                print("REWARD (Player-2):", c.game.score(2), '\n\n', file=self.eval_conversation_file, flush=True)

def game_reward(
        prompts, completions, conversation, train_llm_num, # from current batch
        train_llm, opponent_llm, conversation_file, config # general
    ):
    print("\nComputing rewards", flush=True)
    conversations = [deepcopy(c) for c in conversation]
    for idx, action in enumerate(completions): conversations[idx].turn(action)

    train_llm_num = train_llm_num[0]
    conversation = finish_conversations(conversations, train_llm, opponent_llm, train_llm_num, config)

    rewards = [c.game.score(train_llm_num) for c in conversation]

    print('train conversations', file=conversation_file)
    for c, r in zip(conversation, rewards):
        print("CONVERSATION:\n", c, file=conversation_file)
        print("REWARD:", r, '\n'*3, flush=True, file=conversation_file)

    print("Rewards computed", flush=True)
    return rewards
