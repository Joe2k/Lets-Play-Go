# Let's Play Go

9x9 Go with Pygame GUI and AI — PHYS303/CS486/CS686 HW5.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python play.py
```

Options:
```bash
python play.py --engine mcts              # Use MCTS instead of PUCT
python play.py --model-path <path>.pt     # Use a different CNN model
python play.py --ai-iterations 800        # More search iterations
```

## Test

```bash
python -m pytest tests/ -v
```

## Controls

| Action | Effect |
|---|---|
| Click intersection | Place stone |
| Pass | Pass turn (two passes = game over) |
| End game | Concede (2-click confirm) |
| New game | Reset |
| Black/White terr. | Toggle territory dots |

---

## AI Strategy

### The Big Picture

The AI thinks by playing thousands of "what-if" games from the current board position. It learns which moves lead to winning — not by memorizing openings, but by simulating outcomes.

```
┌─────────────────────────────────────────────────────────┐
│                    How the AI Thinks                     │
│                                                         │
│   Board State ──► CNN (Neural Net) ──► Move Suggestions │
│       │                                    │            │
│       │                                    ▼            │
│       │                           Tree Search           │
│       │                           (PUCT)                │
│       │                                    │            │
│       ▼                                    ▼            │
│   Score Result ◄────────────────── Best Move Selected   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Two AI Modes

| Mode | Full Name | How It Works | Strength |
|---|---|---|---|
| **PUCT** (default) | **P**olicy **U**pper **C**onfidence bounds applied to **T**rees | A CNN tells the search which moves to explore. Smart exploration. | Strong |
| **MCTS** (fallback) | **M**onte **C**arlo **T**ree **S**earch | Plays random games from each position. No neural net needed. | Decent |

### How PUCT Works (Step by Step)

```
┌──────────────────────────────────────────────────────────────┐
│  PUCT Search Loop (runs N times per move)                     │
│                                                              │
│  1. SELECT                                                    │
│     Start at root, walk down the tree picking moves           │
│     that balance "has won before" vs "not tried much"         │
│                                                              │
│  2. EXPAND                                                    │
│     Hit a leaf node? Ask the CNN:                             │
│     - Which moves look good? (policy head)                    │
│     - Who is winning? (value head)                            │
│                                                              │
│  3. BACKUP                                                    │
│     Send the result back up the tree.                         │
│     Every node on the path updates its win rate.              │
│                                                              │
│  4. PICK                                                      │
│     After all iterations, pick the most-visited child.        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### The PUCT Formula (Simplified)

```
Score = Q + c × P × √(parent_visits) / (1 + child_visits)
        │   │   │
        │   │   └── Explores moves the CNN says are good
        │   └── Tuning knob (c_puct)
        └── How often this move has won so far
```

- **Q (value)**: "This move has won X% of the time" — exploitation
- **P (prior)**: "The CNN thinks this move is good" — prior knowledge
- The formula balances trying known-good moves vs exploring new ones

### The CNN (Neural Network)

```
┌──────────────────────────────────────────────┐
│           TinyPolicyValueNet                  │
│                                              │
│  Input: 8 planes × 9 × 9                     │
│  ┌────────────────────────────────────┐      │
│  │ Plane 1-2: Own/Opp stones (now)    │      │
│  │ Plane 3-4: Own/Opp stones (t-1)    │      │
│  │ Plane 5-6: Own/Opp stones (t-2)    │      │
│  │ Plane 7:   Color to move           │      │
│  │ Plane 8:   Legal moves mask        │      │
│  └────────────────────────────────────┘      │
│                    │                         │
│                    ▼                         │
│          Conv → 4× ResBlock                  │
│                    │                         │
│          ┌─────────┼─────────┐               │
│          ▼         ▼         ▼               │
│      Policy     Value   Ownership            │
│      (82)       (1)      (81)                │
│   "which     "who      "who owns             │
│    move"     wins"     each spot"            │
│                                              │
└──────────────────────────────────────────────┘
```

### How Components Talk to Each Other

```
┌──────────┐     ┌──────────────┐     ┌──────────┐
│          │     │              │     │          │
│  GUI     │────►│  Go Engine   │◄───►│  AI      │
│  (app.py)│     │  (go_engine) │     │  (puct)  │
│          │     │              │     │          │
└──────────┘     └──────┬───────┘     └────┬─────┘
                        │                  │
                        │                  │
                        ▼                  ▼
                 ┌──────────────┐   ┌──────────────┐
                 │  Board State │   │  CNN Model   │
                 │  Captures    │   │  (model.py)  │
                 │  Score       │   │  (model_v1)  │
                 └──────────────┘   └──────────────┘
```

1. **GUI** shows the board and captures your clicks
2. **Go Engine** validates moves, handles captures/ko/suicide, scores the game
3. **AI** asks the engine for legal moves, runs PUCT search, picks the best one
4. **CNN** evaluates board positions during search (called thousands of times per move)

---

## Training Pipeline

The CNN was trained through **self-play** — the AI played against itself, learned from the results, and repeated.

```
┌──────────────────────────────────────────────────────────────┐
│                    Training Pipeline                          │
│                                                              │
│  ┌─────────────┐                                            │
│  │  Gen 0:     │  MCTS plays against itself                 │
│  │  Bootstrap  │  Generates training data (selfplay.pt)     │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │  Train      │  CNN learns from self-play data            │
│  │  Network    │  (policy + value + ownership heads)        │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │  Evaluate   │  Does the new model beat the old one?      │
│  │  (Gate)     │  Win rate must exceed threshold            │
│  └──────┬──────┘                                            │
│         │                                                    │
│    ┌────┴────┐                                              │
│    │  Yes?   │──► Promote model, repeat from Gen 1          │
│    │  No?    │──► Skip, try again with more data            │
│    └─────────┘                                              │
│                                                              │
│  Repeat until model is strong enough.                        │
└──────────────────────────────────────────────────────────────┘
```

### Training Details

| Step | What Happens | Script |
|---|---|---|
| **Self-play** | PUCT plays games, records board states + outcomes | `generate_selfplay_data.py` |
| **Augment** | Each board rotated/flipped 8 ways (dihedral symmetry) | Built into training |
| **Train** | CNN learns to predict moves and winners | `train_policy_value.py` |
| **Evaluate** | New model plays old model to check improvement | `eval_match.py` |
| **Promote** | If win rate > threshold, new model becomes "best" | `run_pipeline.py` |

### Data Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Self-Play   │───►│  Train CNN   │───►│  Evaluate    │
│  Generation  │    │  (8x aug)    │    │  (Gate)      │
│              │    │              │    │              │
│  boards +    │    │  policy loss │    │  win rate    │
│  outcomes    │    │  value loss  │    │  vs best     │
│  → .pt file  │    │  ownership   │    │  → pass/fail │
└──────────────┘    │  loss        │    └──────┬───────┘
                    └──────────────┘           │
                                               ▼
                                        ┌──────────────┐
                                        │  Promote?    │
                                        │  Yes → best  │
                                        │  No → retry  │
                                        └──────────────┘
```
