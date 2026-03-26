# play_vs_user.py
import os
import numpy as np
from environment import (
    init_state, available_actions, make_move, check_winner,
    print_board, print_board_with_indices
)

Q_PATH = "qtable.npy"

def load_qtable(path: str = Q_PATH) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Q-table introuvable: {path}\n"
            f"👉 Lance d'abord l'entraînement:  python train_agent.py --episodes 50000"
        )
    Q = np.load(path, allow_pickle=True).item()
    # conversion éventuelle en ndarray float32
    for k, v in list(Q.items()):
        Q[k] = np.array(v, dtype=np.float32)
    return Q

def agent_best_action(Q: dict, state: str) -> int:
    acts = available_actions(state)
    if not acts:
        raise RuntimeError("Aucune action légale.")
    qvals = Q.get(state)
    if qvals is None:
        # état jamais vu: simple fallback (centre > coins > autres)
        if 4 in acts: return 4
        for i in [0,2,6,8]:
            if i in acts: return i
        return acts[0]
    return max(acts, key=lambda a: qvals[a])

def play(starter: str = "human"):
    """
    starter in {"human","agent"} — Agent = 'X', Humain = 'O'
    """
    Q = load_qtable(Q_PATH)
    state = init_state()
    current = "O" if starter == "human" else "X"

    print("🎮 Tic-Tac-Toe — Vous êtes 'O', l'agent est 'X'.")
    print("Indices des cases :")
    print_board_with_indices(init_state())
    print("Grille de départ :")
    print_board(state)

    while True:
        if current == "O":
            acts = available_actions(state)
            move_str = input(f"Votre coup (cases libres {acts}) : ")
            while True:
                if move_str.isdigit() and int(move_str) in acts:
                    move = int(move_str)
                    break
                move_str = input("⛔ Invalide. Essayez encore : ")
            state = make_move(state, move, "O")
            print_board(state)
        else:
            a = agent_best_action(Q, state)
            state = make_move(state, a, "X")
            print(f"🤖 Agent joue: {a}")
            print_board(state)

        w = check_winner(state)
        if w:
            if w == "X":
                print("🏆 L'agent gagne.")
            elif w == "O":
                print("🎉 Vous gagnez, bravo !")
            else:
                print("🤝 Match nul.")
            break

        current = "X" if current == "O" else "O"

if __name__ == "__main__":
    start = input("Qui commence ? (human/agent) [human] : ").strip().lower() or "human"
    if start not in ("human", "agent"):
        start = "human"
    play(start)