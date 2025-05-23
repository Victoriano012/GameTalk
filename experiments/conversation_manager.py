from bs4.exceptions import ParserRejectedMarkup
from utils import autoassign
from copy import deepcopy
from bs4 import BeautifulSoup
import ast
import re

def parse_last(text):
    # Regular expression to find all tags with content
    soup = BeautifulSoup(text, 'html.parser')
    matches = [(tag.name, tag.get_text()) for tag in soup.find_all()]
    
    parsed_text = {}
    for tag, content in matches:
        if tag == "think":
            parsed_text = {}
        parsed_text[tag] = content
    
    if len(parsed_text) < 2:
        raise AssertionError("Format error", text)
    
    return parsed_text


class ConversationPlayer:
    def __init__(self, initial_prompt, player_id, game):
        self.pov = initial_prompt
        self.player_id = player_id
        self.game = game

        self.starting_indices = []
        self.ending_indices = []

        self.parsed_actions = []

    def my_turn(self, parsed_action, intermediate_prompt=None, error=False):
        self.pov += "<|start_header_id|>assistant<|end_header_id|> <think>"
        self.starting_indices.append(len(self.pov))

        curr_play = None

        if error:
            self.pov += parsed_action
            curr_play = "error"

        else:
            self.parsed_actions.append(parsed_action)

            if 'play' in parsed_action:
                curr_play = parsed_action['play']
                
            self.pov += parsed_action['think'] + "</think>\n"
            if 'talk' in parsed_action:
                self.pov += "<talk>" + parsed_action['talk'] + "</talk> \n"
            if 'play' in parsed_action:
                self.pov += "<play>" + parsed_action['play'] + "</play> \n"
            self.pov += "<|eot_id|>"

        self.ending_indices.append(len(self.pov))

        my_kwargs, other_kwargs = self.game.make_move(curr_play, self.player_id)
        if my_kwargs is not None:
            self.pov += intermediate_prompt.format(**my_kwargs)
        return my_kwargs, other_kwargs

    def other_turn(self, parsed_action, intermediate_prompt, other_kwargs):
        if 'talk' in parsed_action:
            self.pov += "<|start_header_id|>user<|end_header_id|>" + parsed_action['talk'].strip()
            if 'play' in parsed_action and self.game.show_moves():
                self.pov += "\n<play>" + parsed_action['play'] + "</play>"
            self.pov += "<|eot_id|>\n"
        if other_kwargs is not None:
            self.pov += intermediate_prompt.format(**other_kwargs)

    def get_talk_intervals(self):
        return list(zip(self.starting_indices, self.ending_indices))

class ConversationManager:
    @autoassign
    def __init__(self, initial_prompt_1, initial_prompt_2, intermediate_prompt, name_1, name_2, Game, **initial_kwargs):
        self.game = Game(name_1, name_2, **initial_kwargs)

        self.player_1 = ConversationPlayer(initial_prompt_1.format(my_name=name_1, other_name=name_2, **initial_kwargs), player_id=name_1, game = self.game)
        self.player_2 = ConversationPlayer(initial_prompt_2.format(my_name=name_2, other_name=name_1, **initial_kwargs), player_id=name_2, game = self.game)

        self.players = (self.player_1, self.player_2)
        self.names = (name_1, name_2)
        self.num_interactions = 0

        self.full_conversation = str(initial_kwargs) + '\n'
        self.all_actions = []
    
    def __len__(self):
        return len(self.full_conversation)

    def __str__(self):
        return "ConversationManager: " + self.full_conversation

    # other player = True -> query the player who has just played
    def get_query(self, other_player=False):
        return self.players[other_player].pov + "<|start_header_id|>assistant<|end_header_id|> <think>"

    def get_player(self, player_num):
        return self.player_1 if player_num%2 == 1 else self.player_2

    def finished(self):
        return self.game.is_finished()

    def turn(self, action):
        self.num_interactions += 1
        self.all_actions.append(action)

        action = action.strip()
        action = action if action.startswith("<think>") else "<think>" + action

        try:
            parsed_action = parse_last(action)
        except (AssertionError, ParserRejectedMarkup) as e:
            self.players[0].my_turn(action, error=True)
            self.full_conversation += self.names[0] + " did a format error:\n" + action + "\n"
            return

        my_kwargs, other_kwargs = self.players[0].my_turn(parsed_action, self.intermediate_prompt)
        self.players[1].other_turn(parsed_action, self.intermediate_prompt, other_kwargs)

        self.full_conversation += self.names[0] + ":\n    <think>" + parsed_action['think'] + "</think>\n"
        if 'talk' in parsed_action:
            self.full_conversation += "    <talk>" + parsed_action['talk'] + "</talk> \n"
        if 'play' in parsed_action:
            self.full_conversation += "    <play>" + parsed_action['play'] + "</play> \n"
        
        if my_kwargs is not None:
            self.full_conversation += self.names[0] + " pov:\n" + self.intermediate_prompt.format(**my_kwargs) + "\n"
        if other_kwargs is not None:
            self.full_conversation += self.names[1] + " pov:\n" + self.intermediate_prompt.format(**other_kwargs) + "\n"

        self.players = (self.players[1], self.players[0])
        self.names = (self.names[1], self.names[0])
    
    def get_subconversations(self, player_num):
        conv = ConversationManager(self.initial_prompt_1, self.initial_prompt_2, self.intermediate_prompt, self.name_1, self.name_2, type(self.game), **self.initial_kwargs)
        for idx, action in enumerate(self.all_actions):
            if conv.finished(): break # can be erased, ?
            if (idx+1) % 2 == player_num % 2: yield deepcopy(conv)
            conv.turn(action)


def conversation_to_manager(conversation_text, initial_prompt_1, initial_prompt_2, intermediate_prompt, name_1, name_2, Game):

    ##### parse_conversation_text #####

    lines = conversation_text.strip().splitlines()

    manager_line = lines.pop(0)
    if manager_line == "CONVERSATION:": manager_line = lines.pop(0).strip()
    kwargs = ast.literal_eval(manager_line.removeprefix('ConversationManager: '))

    turns = []
    current_turn = []
    current_player = None

    for line in lines:
        match = re.match(r"^Player-(1|2):$", line)
        if not match: match = re.match(r"^Player-(1|2) did a format error:$", line)
        pov_match = re.match(r"^Player-(1|2) pov:$", line)

        if pov_match or match:
            if current_player is not None and match:
                assert int(match.group(1)) == 3-current_player, f"Player turns are not alternating correctly. Expected Player-{3-current_player}, got Player-{match.group(1)}." + "\n\n" + conversation_text
            if current_player is not None:
                turns.append('\n'.join(current_turn).strip())
            current_player = int(match.group(1)) if match else None
            current_turn = []
        else: current_turn.append(line)
        
    if current_turn and current_player is not None:
        turns.append('\n'.join(current_turn).strip())

    ####################

    conv = ConversationManager(initial_prompt_1, initial_prompt_2, intermediate_prompt, name_1, name_2, Game, **kwargs)
    for turn in turns: conv.turn(turn)

    return conv
