from bs4 import BeautifulSoup

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
    def __init__(self, initial_prompt, Game):
        self.pov = initial_prompt
        self.Game = Game
        self.play = None

        self.starting_indices = []
        self.ending_indices = []

    def my_turn(self, parsed_action, error=False):
        self.pov += "<|start_header_id|>assistant<|end_header_id|> <think>"
        self.starting_indices.append(len(self.pov))

        if error:
            self.players[0].pov += parsed_action
            self.play = self.Game("error")
            self.ending_indices.append(len(self.pov))
            return
        
        if 'play' in parsed_action:
            self.play = self.Game(parsed_action['play'].lower().strip())
            
        self.pov += parsed_action['think'] + "</think>\n"
        if 'talk' in parsed_action:
            self.pov += "<talk>" + parsed_action['talk'] + "</talk> \n"
        if 'play' in parsed_action:
            self.pov += "<play>" + parsed_action['play'] + "</play> \n"
        self.pov += "<|eot_id|>"

        self.ending_indices.append(len(self.pov))

    def other_turn(self, parsed_action, other_moved_prompt, error=False):
        if error:
            self.play = self.Game.default()
        if 'talk' in parsed_action:
            self.pov += "<|start_header_id|>user<|end_header_id|>" + parsed_action['talk'].strip() + "\n<|eot_id|>"
        if 'play' in parsed_action:
            self.pov += other_moved_prompt


class ConversationManager:
    def __init__(self, initial_prompt, other_moved_prompt, name_1, name_2, Game):
        
        self.player_1 = ConversationPlayer(initial_prompt.format(my_name=name_1, other_name=name_2), Game)
        self.player_2 = ConversationPlayer(initial_prompt.format(my_name=name_2, other_name=name_1), Game)

        self.other_moved_prompt = other_moved_prompt
        self.players = (self.player_1, self.player_2)
        self.names = (name_1, name_2)
        self.num_interactions = 0
        self.Game = Game

        self.full_conversation = ""

    # other player = True -> query the player who has just played
    def get_query(self, other_player=False):
        return self.players[other_player].pov + "<|start_header_id|>assistant<|end_header_id|> <think>"

    def finished(self):
        return self.player_1.play is not None and self.player_2.play is not None

    def get_trainable(self, player_num, interaction_idx):
        player = self.player_1 if player_num == 1 else self.player_2
        att_idx = (player.starting_indices[interaction_idx], player.ending_indices[interaction_idx])
        return player.pov[:att_idx[1]], att_idx[0]


    def turn(self, action):
        self.num_interactions += 1
        try:
            parsed_action = parse_last("<think>" + action)
        except AssertionError as e:
            self.players[0].my_turn(parsed_action, error=True)
            self.players[1].other_turn(parsed_action, self.other_moved_prompt, error=True)
            return

        self.players[0].my_turn(parsed_action)
        self.players[1].other_turn(parsed_action, self.other_moved_prompt)

        if self.players[0].play is None and self.players[1].play is not None:
            self.players[0].play = self.Game("error")

        self.full_conversation += self.names[0] + ":\n    <think>" + parsed_action['think'] + "</think>\n"
        if 'talk' in parsed_action:
            self.full_conversation += "    <talk>" + parsed_action['talk'] + "</talk> \n"
        if 'play' in parsed_action:
            self.full_conversation += "    <play>" + parsed_action['play'] + "</play> \n"
        
        self.players = (self.players[1], self.players[0])
        self.names = (self.names[1], self.names[0])
    
    def get_moves(self):
        return self.player_1.play, self.player_2.play