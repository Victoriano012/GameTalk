"""
This file contains the implementation of games
Each instance of a game is just one move, initialized as Game(llm_move)
    where llm_move is a lowercase string with no space at beginning or end
    llm_move may be an invalid move, in that case self.is_error() = True
Each game must have defined a score method that takes two instances and returns the score (float/double) of the first player
"""

# Returns the game instance based on the game name
def get_game(game_name: str):
    if game_name == "rock-paper-scissors":
        return RPS
    return None


from enum import Enum
from collections import defaultdict

# rock-paper-scissors implementation
class RPS(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    ERROR = "error"
    
    def _missing_(value):
        return RPS.ERROR

    def is_error(self):
        return self == RPS.ERROR
        
    def default():
        return RPS.ROCK
    
    # move1 wins -> 2., move2 wins -> 0., tie -> 1.
    def score(move1, move2) -> int:
        if move1 == None: move1 = RPS.ERROR
        if move2 == None: move2 = RPS.ERROR

        # error cases
        if move1 == RPS.ERROR:
            return -1.
        if move2 == RPS.ERROR:
            return 2.
        
        mapping = {RPS.ROCK: 0, RPS.PAPER: 1, RPS.SCISSORS: 2}
        score = (mapping[move1] - mapping[move2] + 3) % 3
        score = -1 if score == 2 else score
        return float(score+1)
    
    def game_metrics(moves):
        rewards = [RPS.score(w[0], w[1]) for w in moves]
        
        metrics = defaultdict(float)
        metrics["win_rate"] = sum(1 for r in rewards if r == 2.)
        metrics["draw_rate"] = sum(1 for r in rewards if r == 1.)
        metrics["loss_rate"] = sum(1 for r in rewards if r <= 0.)

        metrics["lost_by_error (%)"] += sum(1 for w in moves if w[0].is_error())
        metrics["won_by_error (%)"] += sum(1 for w in moves if w[1].is_error() and not w[0].is_error())

        return metrics
    