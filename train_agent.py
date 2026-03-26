# train_agent.py
"""
Tic-Tac-Toe (X-O) — Entraînement Q-learning de l'agent X contre un adversaire O aléatoire.

- Algorithme : Q-learning tabulaire
- Etat : chaîne de longueur 9 avec ' ', 'X', 'O'
- Actions : indices 0..8 (cases vides)
- Récompenses :
    +1.0 si l'agent (X) gagne
    -1.0 si l'agent perd (O gagne)
    +0.5 si match nul
     0.0 sinon (transition non terminale), avec bootstrap gamma * max_a' Q(s', a')

Sorties :
- qtable.npy  : dictionnaire Python {state_str: np.ndarray shape (9,)} (pour usage Python)
- qtable.json : dictionnaire JSON {state_str: [float,...]} (pour PyScript dans le navigateur)

Utilisation :
    python train_agent.py
    python train_agent.py --episodes 100000 --out qtable.npy
    python train_agent.py --episodes 50000 --seed 42
    python train_agent.py --eval 1000            # évalue après entraînement (facultatif)
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Dict

import numpy as np
from environment import (
    init_state, available_actions, make_move, check_winner
)

# -----------------------------
# Q-table : dict[state_str] -> np.array shape (9,)
# -----------------------------
Q: Dict[str, np.ndarray] = {}


def get_Q(state: str) -> np.ndarray:
    """Retourne le vecteur Q(s, :) ; initialise à 0 si état jamais vu."""
    if state not in Q:
        Q[state] = np.zeros(9, dtype=np.float32)
    return Q[state]


def choose_action(state: str, epsilon: float) -> int:
    """Politique epsilon-greedy, restreinte aux actions légales (cases vides)."""
    acts = available_actions(state)
    if not acts:
        raise RuntimeError("Plus d'actions disponibles (état terminal ou invalide).")
    if random.random() < epsilon:
        return random.choice(acts)
    qvals = get_Q(state)
    # argmax restreint aux actions légales
    return max(acts, key=lambda a: qvals[a])


def train(
    episodes: int = 50_000,
    alpha: float = 0.7,
    gamma: float = 0.95,
    epsilon: float = 0.2,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.999,
    log_every: int = 5_000,
) -> None:
    """
    Entraîne l'agent X contre un adversaire O aléatoire.

    Args:
        episodes: nombre d'épisodes d'entraînement
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration initiale
        epsilon_min: exploration minimale
        epsilon_decay: facteur de décroissance par épisode
        log_every: fréquence d'affichage d'un petit log
    """
    global Q

    for ep in range(episodes):
        state = init_state()

        while True:
            # --- Tour de l'agent (X)
            a = choose_action(state, epsilon)
            s_x = make_move(state, a, "X")
            w = check_winner(s_x)

            if w == "X":
                # Victoire agent
                get_Q(state)[a] += alpha * (1.0 - get_Q(state)[a])
                break
            elif w == "draw":
                # Match nul
                get_Q(state)[a] += alpha * (0.5 - get_Q(state)[a])
                break

            # --- Tour de l'adversaire (O) aléatoire
            opp_actions = available_actions(s_x)
            opp_a = random.choice(opp_actions)
            s_o = make_move(s_x, opp_a, "O")
            w = check_winner(s_o)

            if w == "O":
                # Défaite agent
                get_Q(state)[a] += alpha * (-1.0 - get_Q(state)[a])
                break
            elif w == "draw":
                # Match nul
                get_Q(state)[a] += alpha * (0.5 - get_Q(state)[a])
                break

            # --- Transition non terminale : bootstrap
            future_q = get_Q(s_o)
            legal = available_actions(s_o)
            max_next = max((future_q[i] for i in legal), default=0.0)
            target = 0.0 + gamma * max_next
            get_Q(state)[a] += alpha * (target - get_Q(state)[a])

            # Continuer depuis l'état après le coup adverse
            state = s_o

        # Décroissance epsilon par épisode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if log_every and (ep + 1) % log_every == 0:
            print(f"[Ep {ep+1:>6}/{episodes}] epsilon={epsilon:.3f}")


def evaluate(Q: Dict[str, np.ndarray], games: int = 1000, seed: int | None = None) -> dict:
    """
    Joue un nombre fixe de parties sans exploration (greedy) contre un adversaire aléatoire
    et renvoie les stats : {'win': x, 'draw': y, 'loss': z}.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    def agent_best_action(state: str) -> int:
        acts = available_actions(state)
        qvals = Q.get(state)
        if qvals is None:
            # Fallback simple si état non vu : centre > coins > autres
            if 4 in acts:
                return 4
            for i in [0, 2, 6, 8]:
                if i in acts:
                    return i
            return acts[0]
        return max(acts, key=lambda a: qvals[a])

    stats = {"win": 0, "draw": 0, "loss": 0}

    for _ in range(games):
        state = init_state()
        current = "X"  # l'agent commence pour cette évaluation
        while True:
            if current == "X":
                a = agent_best_action(state)
                state = make_move(state, a, "X")
            else:
                acts = available_actions(state)
                state = make_move(state, random.choice(acts), "O")

            w = check_winner(state)
            if w:
                if w == "X":
                    stats["win"] += 1
                elif w == "O":
                    stats["loss"] += 1
                else:
                    stats["draw"] += 1
                break

            current = "O" if current == "X" else "X"

    return stats


def save_q_npy(Q: Dict[str, np.ndarray], path: str) -> None:
    """Sauvegarde la Q-table au format .npy (dict picklé)."""
    np.save(path, Q, allow_pickle=True)


def save_q_json(Q: Dict[str, np.ndarray], path: str) -> None:
    """Sauvegarde la Q-table en JSON (utilisable par PyScript dans le navigateur)."""
    as_json = {k: [float(x) for x in v] for k, v in Q.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(as_json, f)


def main():
    parser = argparse.ArgumentParser(description="Entraînement Q-learning Tic-Tac-Toe (agent X).")
    parser.add_argument("--episodes", type=int, default=50_000, help="Nombre d'épisodes d'entraînement")
    parser.add_argument("--alpha", type=float, default=0.7, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Exploration initiale")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Exploration minimale")
    parser.add_argument("--epsilon_decay", type=float, default=0.999, help="Décroissance d'epsilon par épisode")
    parser.add_argument("--seed", type=int, default=42, help="Seed pour reproductibilité")
    parser.add_argument("--out", type=str, default="qtable.npy", help="Chemin de sortie .npy")
    parser.add_argument("--json", type=str, default="qtable.json", help="Chemin de sortie .json (PyScript)")
    parser.add_argument("--eval", type=int, default=0, help="Nombre de parties d'évaluation après entraînement (0 pour désactiver)")
    args = parser.parse_args()

    # Reproductibilité
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f" Début entraînement : episodes={args.episodes}, alpha={args.alpha}, gamma={args.gamma}, "
          f"epsilon={args.epsilon}→{args.epsilon_min}, decay={args.epsilon_decay}, seed={args.seed}")

    try:
        train(
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
        )
    except KeyboardInterrupt:
        print("\n  Entraînement interrompu manuellement — sauvegarde de l'état courant...")

    # Sauvegardes
    save_q_npy(Q, args.out)
    save_q_json(Q, args.json)
    print(f" Q-table sauvegardée :\n - {args.out}\n - {args.json}")

    # Evaluation (optionnelle)
    if args.eval and args.eval > 0:
        stats = evaluate(Q, games=args.eval, seed=args.seed)
        total = sum(stats.values())
        w, d, l = stats["win"], stats["draw"], stats["loss"]
        print(f" Evaluation ({total} parties, agent greedy vs random O) : "
              f"Win={w} ({w/total:.1%}) | Draw={d} ({d/total:.1%}) | Loss={l} ({l/total:.1%})")


if __name__ == "__main__":
    main()