from bs4.exceptions import ParserRejectedMarkup
from bs4 import BeautifulSoup
from copy import deepcopy
from utils import autoassign

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

    def my_turn(self, parsed_action, error=False):
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

        self.game.make_move(curr_play, self.player_id)
        self.ending_indices.append(len(self.pov))

    def other_turn(self, parsed_action, other_moved_prompt):
        if 'talk' in parsed_action:
            self.pov += "<|start_header_id|>user<|end_header_id|>" + parsed_action['talk'].strip() + "\n<|eot_id|>"
        if 'play' in parsed_action:
            self.pov += other_moved_prompt


class ConversationManager:
    @autoassign
    def __init__(self, initial_prompt, other_moved_prompt, name_1, name_2, Game, **initial_kwargs):
        self.game = Game(name_1, name_2, **initial_kwargs)

        self.player_1 = ConversationPlayer(initial_prompt.format(my_name=name_1, other_name=name_2, **initial_kwargs), player_id=name_1, game = self.game)
        self.player_2 = ConversationPlayer(initial_prompt.format(my_name=name_2, other_name=name_1, **initial_kwargs), player_id=name_2, game = self.game)

        self.players = (self.player_1, self.player_2)
        self.names = (name_1, name_2)
        self.num_interactions = 0

        self.full_conversation = ""
        self.all_actions = []
    
    def __len__(self):
        return len(self.full_conversation)

    def __str__(self):
        return "ConversationManager:\n" + self.full_conversation

    # other player = True -> query the player who has just played
    def get_query(self, other_player=False):
        return self.players[other_player].pov + "<|start_header_id|>assistant<|end_header_id|> <think>"

    def finished(self):
        return self.game.is_finished()

    def get_trainable(self, player_num, interaction_idx):
        player = self.player_1 if player_num == 1 else self.player_2
        att_idx = (player.starting_indices[interaction_idx], player.ending_indices[interaction_idx])
        return player.pov[:att_idx[1]], att_idx[0]

    def turn(self, action):
        self.num_interactions += 1
        self.all_actions.append(action)
        try:
            parsed_action = parse_last("<think>" + action)
        except (AssertionError, ParserRejectedMarkup) as e:
            self.players[0].my_turn(action, error=True)
            self.full_conversation += self.names[0] + " did a format error:\n" + action + "\n"
            return

        self.players[0].my_turn(parsed_action)
        self.players[1].other_turn(parsed_action, self.other_moved_prompt)

        self.full_conversation += self.names[0] + ":\n    <think>" + parsed_action['think'] + "</think>\n"
        if 'talk' in parsed_action:
            self.full_conversation += "    <talk>" + parsed_action['talk'] + "</talk> \n"
        if 'play' in parsed_action:
            self.full_conversation += "    <play>" + parsed_action['play'] + "</play> \n"
        
        self.players = (self.players[1], self.players[0])
        self.names = (self.names[1], self.names[0])
    
    def get_subconversations(self, player_num):
        conv = ConversationManager(self.initial_prompt, self.other_moved_prompt, self.name_1, self.name_2, type(self.game), **self.initial_kwargs)
        for idx, action in enumerate(self.all_actions):
            if (idx+1) % 2 == player_num % 2:
                yield deepcopy(conv)
            conv.turn(action)
