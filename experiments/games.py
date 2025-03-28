"""
This file contains the implementation of games
Each instance of a game is just one move, initialized as Game(llm_move)
    where llm_move is a lowercase string with no space at beginning or end
    if llm_move may an invalid move, in that case self.is_error() = True
Each game must have defined a score method that takes two instances and returns the score (float/double) of the first player
"""

# Returns the game instance based on the game name
def get_game(game_name: str):
    if game_name == "rock-paper-scissors":
        return RPS
    return None


from enum import Enum

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
    
    # move1 wins -> 2., move2 wins -> 0., tie -> 1.
    def score(move1, move2) -> int:

        # error cases
        if move1 == RPS.ERROR:
            return 0.
        if move2 == RPS.ERROR:
            return 2.
        
        mapping = {RPS.ROCK: 0, RPS.PAPER: 1, RPS.SCISSORS: 2}
        score = (mapping[move1] - mapping[move2] + 3) % 3
        score = -1 if score == 2 else score
        return float(score+1)