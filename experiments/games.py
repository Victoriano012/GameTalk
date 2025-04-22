from types import SimpleNamespace

# Returns the game instance based on the game name
def get_game(game_name: str):
    if game_name == "rock-paper-scissors":
        return RPS
    return None


from enum import Enum
from collections import defaultdict

# rock-paper-scissors implementation
class RPS():
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    ERROR = "error"

    def __init__(self, id_1, id_2):
        self.player_1 = SimpleNamespace(id=id_1, move=None)
        self.player_2 = SimpleNamespace(id=id_2, move=None)

        self.ids = (id_1, id_2)

    def make_move(self, move, player_id):
        move = move.strip().lower()
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        curr_player, other_player = (self.player_1, self.player_2) if player_id == self.player_1.id else (self.player_2, self.player_1)

        if move is None:
            if other_player.move is not None:
                curr_player.move = RPS.ERROR
        elif move not in (RPS.ROCK, RPS.PAPER, RPS.SCISSORS):
            curr_player.move = RPS.ERROR
        else:
            curr_player.move = move
    
    # move1 wins -> 2., move2 wins -> 0., tie -> 1.
    def score(self, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        move1, move2 = (self.player_1.move, self.player_2.move) if player_id == self.player_1.id else (self.player_2.move, self.player_1.move)

        # error cases
        if move1 == RPS.ERROR or move1 == None: return -1.
        if move2 == RPS.ERROR or move2 == None: return 2.
        
        mapping = {RPS.ROCK: 0, RPS.PAPER: 1, RPS.SCISSORS: 2}
        score = (mapping[move1] - mapping[move2] + 3) % 3
        score = -1 if score == 2 else score
        return float(score+1)
        
    def won_by_error(self, player_id):
        if isinstance(player_id, int):
            player_id = self.player_1.id if player_id%2 == 1 else self.player_2.id
        if player_id not in self.ids:
            raise ValueError("Invalid player ID")
        
        move1, move2 = (self.player_1.move, self.player_2.move) if player_id == self.player_1.id else (self.player_2.move, self.player_1.move)

        if move1 == RPS.ERROR or move1 == None: return False
        return move2 == RPS.ERROR or move2 == None
    
    def game_metrics(games, player_id):
        rewards = [g.score(player_id) for g in games]
        
        metrics = defaultdict(float)
        metrics["win_rate"] = sum(1 for r in rewards if r == 2.)
        metrics["draw_rate"] = sum(1 for r in rewards if r == 1.)
        metrics["loss_rate"] = sum(1 for r in rewards if r <= 0.)

        metrics["lost_by_error (%)"] += sum(1 for r in rewards if r == -1.)
        metrics["won_by_error (%)"] += sum(1 for g in games if g.won_by_error(player_id))

        return metrics
    
    def is_finished(move1, move2):
        return move1 is not None and move2 is not None
