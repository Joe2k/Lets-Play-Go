"""Entry point for 9x9 Go GUI vs PUCT AI.

AI Strategy
-----------
PUCT (Policy UCT) — a neural network guides the tree search. A small
ResNet trained on self-play games predicts which moves are promising
(policy) and who is winning (value). The search explores promising moves
more deeply and avoids wasting time on bad ones. Much stronger than
random rollouts. MCTS is also available as a fallback option.

Design Decisions
----------------
- Flat list of 81 ints for the board (faster than nested lists)
- Group caching with lazy rebuild for efficient liberty counting
- 8-plane NN input: own/opp stones at t, t-1, t-2, color, legal mask
- Batched NN inference for GPU efficiency during self-play
- clone_fast() for search hot paths (avoids deepcopy overhead)
- Benson's algorithm for conservative dead-stone detection
- Key challenge: rule order in place_stone (capture first, then suicide,
  then ko) — getting this wrong breaks snapbacks and ko detection

Testing
-------
- 47 unit tests: engine rules (placement, capture, ko, suicide, scoring),
  MCTS behavior (no eye-filling, tree reuse, RAVE stats), PUCT behavior,
  and GNU Go integration
- Self-play stress tests (full games complete without crash)
- Head-to-head evaluation: 97 logged games vs GNU Go in eval_results.csv
"""

from gui.app import main

if __name__ == "__main__":
    main()
