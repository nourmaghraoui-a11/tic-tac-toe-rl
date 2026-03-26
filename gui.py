# gui.py
import os
import sys
import numpy as np
import pygame
from environment import (
    init_state, available_actions, make_move, check_winner
)

Q_PATH = "qtable.npy"

def load_qtable(path: str = Q_PATH) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        print(f"Q-table introuvable ({path}). Lance d'abord: python train_agent.py")
        sys.exit(1)
    Q = np.load(path, allow_pickle=True).item()
    for k, v in list(Q.items()):
        Q[k] = np.array(v, dtype=np.float32)
    return Q

def agent_best_action(Q: dict, state: str) -> int:
    acts = available_actions(state)
    if not acts:
        raise RuntimeError("Aucune action légale.")
    qvals = Q.get(state)
    if qvals is None:
        if 4 in acts: return 4
        for i in [0,2,6,8]:
            if i in acts: return i
        return acts[0]
    return max(acts, key=lambda a: qvals[a])

# --- Pygame UI ---
WIDTH = 480
HEIGHT = 480
GRID_SIZE = 3
CELL = 120
MARGIN = 30
LINE_COLOR = (30, 30, 30)
BG = (245, 245, 245)
X_COLOR = (40, 110, 230)
O_COLOR = (230, 60, 60)
TXT = (20, 20, 20)
MSG = (10, 130, 70)

def draw_grid(screen):
    # Lignes verticales
    for i in range(1, GRID_SIZE):
        x = MARGIN + i * CELL
        pygame.draw.line(screen, LINE_COLOR, (x, MARGIN), (x, MARGIN + GRID_SIZE * CELL), 4)
    # Lignes horizontales
    for i in range(1, GRID_SIZE):
        y = MARGIN + i * CELL
        pygame.draw.line(screen, LINE_COLOR, (MARGIN, y), (MARGIN + GRID_SIZE * CELL, y), 4)

def draw_marks(screen, state, font):
    for idx, ch in enumerate(state):
        r = idx // 3
        c = idx % 3
        x = MARGIN + c * CELL + CELL // 2
        y = MARGIN + r * CELL + CELL // 2
        if ch == "X":
            text = font.render("X", True, X_COLOR)
            rect = text.get_rect(center=(x, y))
            screen.blit(text, rect)
        elif ch == "O":
            text = font.render("O", True, O_COLOR)
            rect = text.get_rect(center=(x, y))
            screen.blit(text, rect)

def pos_to_index(mx, my):
    if not (MARGIN <= mx < MARGIN + 3 * CELL and MARGIN <= my < MARGIN + 3 * CELL):
        return None
    c = (mx - MARGIN) // CELL
    r = (my - MARGIN) // CELL
    return int(r * 3 + c)

def render_message(screen, small_font, msg, color=MSG):
    text = small_font.render(msg, True, color)
    rect = text.get_rect(center=(WIDTH // 2, HEIGHT - 30))
    screen.blit(text, rect)

def main(starter="human"):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tic-Tac-Toe RL")
    font = pygame.font.SysFont(None, 110)
    small = pygame.font.SysFont(None, 26)

    Q = load_qtable(Q_PATH)

    def reset_game():
        return init_state(), ("O" if starter == "human" else "X"), None

    state, current, ended = reset_game()

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(BG)
        draw_grid(screen)
        draw_marks(screen, state, font)

        if ended is None:
            render_message(screen, small, f"Tour: {current}  —  'R' pour recommencer, 'Esc' pour quitter")
        else:
            if ended == "X":
                render_message(screen, small, "Fin : L'agent (X) gagne  —  'R' pour rejouer")
            elif ended == "O":
                render_message(screen, small, "Fin : Vous gagnez (O)  —  'R' pour rejouer")
            else:
                render_message(screen, small, "Fin : NUL — 'R' pour rejouer")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    state, current, ended = reset_game()

            elif event.type == pygame.MOUSEBUTTONDOWN and ended is None:
                if current == "O":  # humain
                    mx, my = pygame.mouse.get_pos()
                    idx = pos_to_index(mx, my)
                    if idx is not None and state[idx] == " ":
                        state = make_move(state, idx, "O")
                        w = check_winner(state)
                        if w:
                            ended = w
                        else:
                            current = "X"

        # Tour de l'agent (X)
        if ended is None and current == "X":
            pygame.time.delay(250)  # petite pause pour lisibilité
            a = agent_best_action(Q, state)
            state = make_move(state, a, "X")
            w = check_winner(state)
            ended = w if w else None
            if ended is None:
                current = "O"

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    start = input("Qui commence ? (human/agent) [human] : ").strip().lower() or "human"
    if start not in ("human", "agent"):
        start = "human"
    main(starter=start)