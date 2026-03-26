# environment.py
from typing import List, Optional

# État = string de longueur 9, ex: "X O  X   "
# index: 0 1 2
#        3 4 5
#        6 7 8

def init_state() -> str:
    return " " * 9

def available_actions(state: str) -> List[int]:
    return [i for i in range(9) if state[i] == " "]

def make_move(state: str, action: int, player: str) -> str:
    assert player in ("X", "O")
    if state[action] != " ":
        raise ValueError("Case déjà occupée.")
    s = list(state)
    s[action] = player
    return "".join(s)

def check_winner(s: str) -> Optional[str]:
    """Retourne 'X', 'O', 'draw' ou None si la partie continue."""
    wins = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a,b,c in wins:
        if s[a] != " " and s[a] == s[b] == s[c]:
            return s[a]
    if " " not in s:
        return "draw"
    return None

def print_board(state: str) -> None:
    """Affiche la grille simple."""
    for i in range(0, 9, 3):
        a, b, c = state[i:i+3]
        print(f"{a or ' '} | {b or ' '} | {c or ' '}")
        if i < 6:
            print("--+---+--")
    print()

def print_board_with_indices(state: str) -> None:
    """Affiche la grille + indices pour aider l'utilisateur."""
    def cell(i):
        return state[i] if state[i] != " " else str(i)
    for i in range(0, 9, 3):
        a, b, c = cell(i), cell(i+1), cell(i+2)
        print(f"{a} | {b} | {c}")
        if i < 6:
            print("--+---+--")
    print()