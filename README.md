# Hybrid Chess AI Engine

an experimental Chess AI that combines **Classical Alpha-Beta Search** with **Reinforcement Learning** (Temporal Difference Learning). The engine uses traditional piece-square tables and move ordering but refines its evaluations over time by building a persistent value network of chess positions.

## 🚀 Features

* **Hybrid Evaluation:** Blends classical heuristics (material, mobility, king safety) with a learned `ValueNetwork` based on game experience.
* **Search Optimizations:** Implements Iterative Deepening, Quiescence Search, Transposition Tables (Zobrist Hashing), and Aspiration Windows.
* **Opening Book:** Includes a basic library of standard openings (Italian, Sicilian, QG, etc.) to ensure varied gameplay.
* **Self-Play Training:** Supports a "Watch" mode where two AI instances play against each other to populate an Experience Replay buffer and update position values.
* **Visualization:** Ability to export the search tree to a graph image to analyze how the AI is "thinking."

## 🛠️ Components

* `logic.py`: The core engine, including the `ChessAI`, `ValueNetwork`, and `ChessEvaluator`.
* `gui.py`: A Pygame-based interface for playing against the AI or watching self-play.

## 🚦 Getting Started

### Prerequisites

* Python 3.8+
* `python-chess`
* `pygame`
* `numpy`
* `networkx` & `matplotlib` (for search tree visualization)

### Installation

```bash
pip install chess pygame numpy networkx matplotlib

```

### Usage

Run the GUI to start a game or watch the AI train:

```bash
python gui.py

```

## 🎮 Controls

* **H**: Get a move hint from the AI.
* **V**: Visualize the current search tree (saves as `search_tree.png`).
* **R**: Reset the game.
* **N**: Print network statistics to the console.
* **ESC**: Quit.
