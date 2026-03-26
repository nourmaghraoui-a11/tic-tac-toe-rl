# Tic‑Tac‑Toe RL — Démo (PyScript)

Cliquez sur une case vide pour jouer. Vous êtes **O**, l’agent est **X**.

<link rel="stylesheet" href="https://pyscript.net/releases/2024.1.1/pyscript.css" />
<script defer src="https://pyscript.net/releases/2024.1.1/pyscript.js"></script>

<style>
.board { display:grid; grid-template-columns: repeat(3, 100px); gap:10px; margin:10px 0 12px;}
.cell { width:100px; height:100px; font-size:56px; font-weight:700; background:#f4f6fb; border:2px solid #dbe1f1; border-radius:12px; cursor:pointer; display:flex; align-items:center; justify-content:center; }
.cell:disabled { opacity: .7; cursor: default; }
.msg { font-weight:600; color:#2c7d59; }
</style>

<div id="msg" class="msg">Chargement de l’agent…</div>
<div class="board">
  <button class="cell" id="cell-0" py-click="play_cell(0)">0</button>
  <button class="cell" id="cell-1" py-click="play_cell(1)">1</button>
  <button class="cell" id="cell-2" py-click="play_cell(2)">2</button>
  <button class="cell" id="cell-3" py-click="play_cell(3)">3</button>
  <button class="cell" id="cell-4" py-click="play_cell(4)">4</button>
  <button class="cell" id="cell-5" py-click="play_cell(5)">5</button>
  <button class="cell" id="cell-6" py-click="play_cell(6)">6</button>
  <button class="cell" id="cell-7" py-click="play_cell(7)">7</button>
  <button class="cell" id="cell-8" py-click="play_cell(8)">8</button>
</div>

<button py-click="reset_game('human')">Recommencer (Humain)</button>
<button py-click="reset_game('agent')">Recommencer (Agent)</button>

<py-config>
  packages = ["numpy"]
</py-config>

<!-- Si tu mets environment.py à côté, décommente cette ligne : -->
<!-- <py-script src="environment.py"></py-script> -->

<py-script>
from pyscript import Element
from js import document
import numpy as np
import asyncio, json
from pyodide.http import pyfetch

# Fonctions minimales (si environment.py n'est pas importé)
def init_state(): return " " * 9
def available_actions(state): return [i for i in range(9) if state[i] == " "]
def make_move(state, action, player):
    s = list(state); s[action] = player; return "".join(s)
def check_winner(s):
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if s[a] != " " and s[a] == s[b] == s[c]: return s[a]
    return "draw" if " " not in s else None

async def load_qtable_json(url="qtable.json"):
    resp = await pyfetch(url)
    data = await resp.json()
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}

Q = None
state = init_state()
current = "O"     # Humain
ended = None

def write_msg(txt): Element("msg").write(txt)

def render():
    for i in range(9):
        ch = state[i]
        el = document.getElementById(f"cell-{i}")
        el.disabled = (ended is not None) or (ch != " ")
        el.innerText = ch if ch != " " else str(i)
    if ended is None:
        write_msg(f"Tour : {current} — Cliquez sur une case vide.")
    else:
        write_msg("Fin : " + ("🏆 Agent (X) gagne" if ended=="X" else "🎉 Vous gagnez" if ended=="O" else "🤝 Nul"))

def agent_best_action(Q, state):
    acts = available_actions(state)
    qvals = Q.get(state)
    if qvals is None:
        if 4 in acts: return 4
        for i in [0,2,6,8]:
            if i in acts: return i
        return acts[0]
    return max(acts, key=lambda a: qvals[a])

def step_agent():
    global state, current, ended
    if ended or current != "X": return
    a = agent_best_action(Q, state)
    state = make_move(state, a, "X")
    w = check_winner(state)
    ended = w if w else None
    if not ended: current = "O"
    render()

def play_cell(i: int):
    global state, current, ended
    if ended or current != "O": return
    if i not in available_actions(state): return
    state = make_move(state, i, "O")
    w = check_winner(state)
    ended = w if w else None
    if not ended:
        current = "X"
        asyncio.create_task(delayed_agent())
    render()

async def delayed_agent():
    await asyncio.sleep(0.25)
    step_agent()

def reset_game(starter="human"):
    global state, current, ended
    state = init_state()
    current = "O" if starter=="human" else "X"
    ended = None
    render()
    if current == "X":
        asyncio.create_task(delayed_agent())

async def boot():
    global Q
    write_msg("Chargement de l’agent…")
    Q = await load_qtable_json("qtable.json")  # adapte le chemin si nécessaire
    write_msg("Agent chargé. À vous de jouer !")
    render()
    if current == "X":
        await asyncio.sleep(0.25)
        step_agent()

await boot()
</py-script>
