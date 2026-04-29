"""Head-to-head evaluation for MCTS / PUCT agent configs.

Runs multiple games between two agents (alternating colors) and reports:
- wins / win rate
- average score margin
- move latency statistics

All runs append per-game rows to eval_results.csv (overridable via --csv).
Each row records both sides' engine, iterations, and model path so multiple
runs against different checkpoints can be sliced apart later.

Use --a-engine / --b-engine to mix MCTS and PUCT, e.g.:

    # PUCT(net_new) vs PUCT(net_old) — promotion gate.
    python3 scripts/eval_match.py --games 40 \\
        --a-name new --a-engine puct --a-model checkpoints/pv_new.pt --a-iters 200 \\
        --b-name old --b-engine puct --b-model checkpoints/pv_old.pt --b-iters 200

    # PUCT(net) vs MCTS baseline.
    python3 scripts/eval_match.py --games 40 \\
        --a-name puct --a-engine puct --a-model checkpoints/pv_latest.pt --a-iters 200 \\
        --b-name mcts --b-engine mcts --b-iters 400
"""

from __future__ import annotations

import argparse
import csv
import datetime
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.agent import MCTSAgent
from engine.go_engine import BLACK, EMPTY, SIZE, WHITE, GoGame

DEFAULT_CSV = "eval_results.csv"
DEFAULT_MD_SUBDIR = "eval_boards"
# Go-convention column letters skip "I".
_COL_LETTERS = "ABCDEFGHJ"
CSV_COLUMNS = [
    "timestamp",
    "match_id",
    "game_index",
    "black_name",
    "white_name",
    "winner_name",
    "black_engine",
    "white_engine",
    "black_iters",
    "white_iters",
    "black_model",
    "white_model",
    "black_score",
    "white_score",
    "margin_for_black",
    "moves",
    "finished",
    "black_avg_ms",
    "white_avg_ms",
    # Cross-check from GNU Go's final_score (only filled when one side is
    # gnugo and the game ended via two passes). "" otherwise.
    "gnu_winner_name",
    "gnu_margin_for_black",
    "score_match",
]


@dataclass
class AgentSpec:
    name: str
    iterations: int
    engine: str = "mcts"
    model_path: Optional[str] = None
    c_puct: float = 1.25
    add_root_noise: bool = False


def _build_agent(spec: AgentSpec, seed: int):
    if spec.engine == "puct":
        from ai.puct_agent import PUCTAgent
        return PUCTAgent(
            iterations=spec.iterations,
            model_path=spec.model_path,
            seed=seed,
            c_puct=spec.c_puct,
            add_root_noise=spec.add_root_noise,
        )
    if spec.engine == "gnugo":
        from ai.gnugo_agent import GnuGoAgent
        return GnuGoAgent(level=spec.iterations)
    return MCTSAgent(iterations=spec.iterations, seed=seed)


@dataclass
class GameResult:
    black_spec: AgentSpec
    white_spec: AgentSpec
    winner_name: str
    black_score: float
    white_score: float
    margin_for_black: float
    moves: int
    finished: bool
    black_avg_ms: float
    white_avg_ms: float
    # GNU Go cross-check: "" / NaN / "" when not applicable, otherwise
    # populated from final_score on a gnugo participant.
    gnu_winner_name: str = ""
    gnu_margin_for_black: Optional[float] = None
    score_match: Optional[bool] = None
    # Final-position snapshot for the markdown report. dead_* are only
    # non-empty when the game ended via two passes (Benson path); for
    # concession / move-cap finishes they stay empty.
    final_board: list[int] = field(default_factory=list)
    dead_black: set[int] = field(default_factory=set)
    dead_white: set[int] = field(default_factory=set)


def _play_one_game(
    black_spec: AgentSpec,
    white_spec: AgentSpec,
    seed: int,
    max_moves: int,
) -> GameResult:
    game = GoGame()
    black_agent = _build_agent(black_spec, seed)
    white_agent = _build_agent(white_spec, seed + 1)

    black_latencies: list[float] = []
    white_latencies: list[float] = []
    move_count = 0

    while not game.finished and move_count < max_moves:
        is_black_turn = game.to_move == BLACK
        agent = black_agent if is_black_turn else white_agent
        t0 = time.perf_counter()
        move = agent.select_move(game)
        dt = (time.perf_counter() - t0) * 1000.0
        if is_black_turn:
            black_latencies.append(dt)
        else:
            white_latencies.append(dt)

        if move == "pass":
            game.pass_turn()
        else:
            ok = game.place_stone(*move)
            if not ok:
                raise RuntimeError(
                    f"Illegal move from {black_spec.name if is_black_turn else white_spec.name}: {move}"
                )
        move_count += 1

    s = game.score()
    winner_name = black_spec.name if s["winner"] == BLACK else white_spec.name

    # Mirror the engine's dead-stone path purely for markdown reporting:
    # only when the game ended via two passes (same gate as score()).
    dead_black: set[int] = set()
    dead_white: set[int] = set()
    if game.finished and game.loser is None:
        alive_b = game._benson_alive(BLACK)
        alive_w = game._benson_alive(WHITE)
        dead_black, dead_white = game._dead_chains(alive_b, alive_w)

    gnu_winner_name = ""
    gnu_margin_for_black: Optional[float] = None
    score_match: Optional[bool] = None
    # Only meaningful when a gnugo agent is in play AND the game ended
    # via two consecutive passes — concession or move-cap termination
    # leave dead stones / unsettled groups that final_score can't judge.
    from ai.gnugo_agent import GnuGoAgent
    gnu_agent = next(
        (ag for ag in (black_agent, white_agent) if isinstance(ag, GnuGoAgent)),
        None,
    )
    if (gnu_agent is not None
            and game.finished
            and len(game.history) >= 2
            and game.history[-1] == "pass"
            and game.history[-2] == "pass"):
        gnu_score = gnu_agent.final_score(game)
        if gnu_score is not None and gnu_score["winner"] is not None:
            gw = gnu_score["winner"]
            gnu_winner_name = black_spec.name if gw == BLACK else white_spec.name
            gnu_margin_for_black = (
                gnu_score["margin"] if gw == BLACK else -gnu_score["margin"]
            )
            score_match = (gw == s["winner"])

    return GameResult(
        black_spec=black_spec,
        white_spec=white_spec,
        winner_name=winner_name,
        black_score=s["black"],
        white_score=s["white"],
        margin_for_black=s["black"] - s["white"],
        moves=move_count,
        finished=game.finished,
        black_avg_ms=(statistics.mean(black_latencies) if black_latencies else 0.0),
        white_avg_ms=(statistics.mean(white_latencies) if white_latencies else 0.0),
        gnu_winner_name=gnu_winner_name,
        gnu_margin_for_black=gnu_margin_for_black,
        score_match=score_match,
        final_board=list(game.board),
        dead_black=dead_black,
        dead_white=dead_white,
    )


def _board_md_table(
    board: list[int], dead_black: set[int], dead_white: set[int]
) -> str:
    """Render the final board as a markdown table.

    Cells: `B`/`W` for live stones, `b`/`w` for dead (will be removed
    when scoring), `.` for empty. Rows are labeled 9..1 top-to-bottom
    and columns A..J skipping I, matching standard Go notation.
    """
    header = "|   | " + " | ".join(_COL_LETTERS) + " |"
    sep = "|---|" + "|".join(["---"] * SIZE) + "|"
    lines = [header, sep]
    for r in range(SIZE):
        row_label = SIZE - r
        cells: list[str] = []
        for c in range(SIZE):
            idx = r * SIZE + c
            v = board[idx]
            if v == BLACK:
                cells.append("b" if idx in dead_black else "B")
            elif v == WHITE:
                cells.append("w" if idx in dead_white else "W")
            else:
                cells.append(".")
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_md_header(
    md_path: pathlib.Path, match_id: str, a: AgentSpec, b: AgentSpec
) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w") as f:
        f.write(f"# Match: {match_id}\n\n")
        for spec in (a, b):
            f.write(
                f"- **{spec.name}** — engine={spec.engine} "
                f"iters={spec.iterations} model={spec.model_path or '-'}\n"
            )
        f.write(
            "\nLegend: `B`/`W` = live stone, `b`/`w` = dead "
            "(scored as opponent territory), `.` = empty.\n\n"
        )


def _append_md_game(
    md_path: pathlib.Path, game_idx: int, r: GameResult
) -> None:
    table = _board_md_table(r.final_board, r.dead_black, r.dead_white)
    finished_kind = "finished" if r.finished else "move cap (not finished)"
    dead_count = len(r.dead_black) + len(r.dead_white)
    with md_path.open("a") as f:
        f.write(
            f"## Game {game_idx + 1}: {r.black_spec.name} (B) vs "
            f"{r.white_spec.name} (W)\n\n"
        )
        f.write(
            f"- **Winner:** {r.winner_name}\n"
            f"- **Score:** B={r.black_score:.1f}  W={r.white_score:.1f} "
            f"(margin for B: {r.margin_for_black:+.1f})\n"
            f"- **Moves:** {r.moves}  ·  **End:** {finished_kind}\n"
        )
        if dead_count:
            f.write(
                f"- **Dead stones removed:** "
                f"black={len(r.dead_black)}  white={len(r.dead_white)}\n"
            )
        if r.score_match is True:
            f.write(
                f"- **GNU Go cross-check:** agreed "
                f"(gnu margin for B = {r.gnu_margin_for_black:+.1f})\n"
            )
        elif r.score_match is False:
            f.write(
                f"- **GNU Go cross-check:** DISAGREED — "
                f"gnu winner={r.gnu_winner_name}, "
                f"gnu margin for B = {r.gnu_margin_for_black:+.1f}\n"
            )
        f.write("\n" + table + "\n\n")


def _check_csv_schema(csv_path: pathlib.Path) -> None:
    """Refuse to append to an existing CSV with a different header."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
    if header != CSV_COLUMNS:
        raise SystemExit(
            f"\nrefusing to append: {csv_path} has a different schema.\n"
            f"  existing header: {header}\n"
            f"  expected header: {CSV_COLUMNS}\n"
            f"  fix: delete the file (or pass --csv <new_path>) and re-run."
        )


def _append_csv(
    csv_path: pathlib.Path,
    rows: list[GameResult],
    timestamp: str,
    match_id: str,
    start_index: int = 0,
) -> None:
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_COLUMNS)
        for offset, r in enumerate(rows):
            score_match_cell = (
                "" if r.score_match is None else int(r.score_match)
            )
            gnu_margin_cell = (
                "" if r.gnu_margin_for_black is None
                else f"{r.gnu_margin_for_black:.2f}"
            )
            w.writerow(
                [
                    timestamp,
                    match_id,
                    start_index + offset,
                    r.black_spec.name,
                    r.white_spec.name,
                    r.winner_name,
                    r.black_spec.engine,
                    r.white_spec.engine,
                    r.black_spec.iterations,
                    r.white_spec.iterations,
                    r.black_spec.model_path or "",
                    r.white_spec.model_path or "",
                    f"{r.black_score:.2f}",
                    f"{r.white_score:.2f}",
                    f"{r.margin_for_black:.2f}",
                    r.moves,
                    int(r.finished),
                    f"{r.black_avg_ms:.2f}",
                    f"{r.white_avg_ms:.2f}",
                    r.gnu_winner_name,
                    gnu_margin_cell,
                    score_match_cell,
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate two agent configs head-to-head")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--max-moves", type=int, default=220)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument("--a-iters", type=int, default=400)
    p.add_argument("--b-iters", type=int, default=400)
    p.add_argument("--a-engine", choices=["mcts", "puct", "gnugo"], default="mcts")
    p.add_argument("--b-engine", choices=["mcts", "puct", "gnugo"], default="mcts")
    p.add_argument("--a-model", type=str, default=None)
    p.add_argument("--b-model", type=str, default=None)
    p.add_argument("--csv", default=DEFAULT_CSV,
                   help=f"CSV output path (default: {DEFAULT_CSV}). Pass empty string to disable.")
    p.add_argument("--md", default=None,
                   help=("Markdown report path. Default: auto-derived as "
                         f"<csv_dir>/{DEFAULT_MD_SUBDIR}/<match_id>.md. "
                         "Pass empty string to disable."))
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="If set, exit code 0 only when A's win rate >= this value (e.g. 0.55).")
    p.add_argument("--c-puct", type=float, default=1.25,
                   help="Exploration constant for PUCT (lower = more exploitation).")
    p.add_argument("--add-noise", action="store_true",
                   help="Enable Dirichlet root noise for PUCT (off by default — "
                        "evaluation should not inject exploration noise).")
    args = p.parse_args()

    a = AgentSpec(name=args.a_name, iterations=args.a_iters,
                  engine=args.a_engine, model_path=args.a_model,
                  c_puct=args.c_puct, add_root_noise=args.add_noise)
    b = AgentSpec(name=args.b_name, iterations=args.b_iters,
                  engine=args.b_engine, model_path=args.b_model,
                  c_puct=args.c_puct, add_root_noise=args.add_noise)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    match_id = f"{a.name}_vs_{b.name}_{timestamp}"

    csv_path: Optional[pathlib.Path] = None
    if args.csv:
        csv_path = pathlib.Path(args.csv)
        # Touch parent dir if user passed something like out/eval.csv.
        if csv_path.parent and not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        _check_csv_schema(csv_path)

    md_path: Optional[pathlib.Path] = None
    if args.md is None:
        md_root = csv_path.parent if csv_path is not None else pathlib.Path(".")
        md_path = md_root / DEFAULT_MD_SUBDIR / f"{match_id}.md"
    elif args.md != "":
        md_path = pathlib.Path(args.md)
    if md_path is not None:
        _write_md_header(md_path, match_id, a, b)

    print(f"match: {match_id}")
    print(f"  {a.name}: engine={a.engine} iters={a.iterations} model={a.model_path or '-'}")
    print(f"  {b.name}: engine={b.engine} iters={b.iterations} model={b.model_path or '-'}")
    print(f"  csv: {csv_path or '(disabled)'}")
    print(f"  md:  {md_path or '(disabled)'}")

    results: list[GameResult] = []
    started = time.perf_counter()
    a_wins_running = 0
    b_wins_running = 0

    try:
        for i in range(args.games):
            if i % 2 == 0:
                black_spec, white_spec = a, b
            else:
                black_spec, white_spec = b, a

            game_started = time.perf_counter()
            result = _play_one_game(
                black_spec=black_spec,
                white_spec=white_spec,
                seed=args.seed + 1009 * i,
                max_moves=args.max_moves,
            )
            game_dt = time.perf_counter() - game_started
            results.append(result)

            if result.winner_name == a.name:
                a_wins_running += 1
            else:
                b_wins_running += 1

            played = i + 1
            wr = a_wins_running / played * 100.0
            score_note = ""
            if result.score_match is False:
                score_note = (
                    f" [score MISMATCH: ours={result.winner_name}"
                    f" ({result.margin_for_black:+.1f}) vs"
                    f" gnu={result.gnu_winner_name}"
                    f" ({result.gnu_margin_for_black:+.1f})]"
                )
            print(
                f"  [{played}/{args.games}] {result.black_spec.name}(B) vs {result.white_spec.name}(W) "
                f"-> {result.winner_name} | score {result.black_score:.1f}-{result.white_score:.1f} "
                f"| {result.moves} moves | {game_dt:.1f}s | "
                f"{a.name}={a_wins_running} {b.name}={b_wins_running} ({wr:.1f}%)"
                f"{score_note}"
            )

            if csv_path is not None:
                _append_csv(csv_path, [result], timestamp, match_id, start_index=i)
            if md_path is not None:
                _append_md_game(md_path, i, result)
    except KeyboardInterrupt:
        print("\nInterrupted; reporting on completed games.")

    elapsed = time.perf_counter() - started
    if not results:
        print("No games completed.")
        sys.exit(1 if args.gate_threshold is not None else 0)

    a_wins = sum(1 for r in results if r.winner_name == a.name)
    b_wins = len(results) - a_wins
    a_winrate = a_wins / len(results)
    avg_moves = statistics.mean(r.moves for r in results)
    avg_margin_for_black = statistics.mean(r.margin_for_black for r in results)

    margins_for_a: list[float] = []
    for r in results:
        if r.black_spec.name == a.name:
            margins_for_a.append(r.margin_for_black)
        else:
            margins_for_a.append(-r.margin_for_black)
    avg_margin_for_a = statistics.mean(margins_for_a)

    a_move_latencies: list[float] = []
    b_move_latencies: list[float] = []
    for r in results:
        if r.black_spec.name == a.name:
            a_move_latencies.append(r.black_avg_ms)
            b_move_latencies.append(r.white_avg_ms)
        else:
            a_move_latencies.append(r.white_avg_ms)
            b_move_latencies.append(r.black_avg_ms)

    print()
    print(f"Games: {len(results)} in {elapsed:.2f}s")
    print(f"{a.name} ({a.engine}, iters={a.iterations}) wins: {a_wins} ({a_winrate*100:.1f}%)")
    print(f"{b.name} ({b.engine}, iters={b.iterations}) wins: {b_wins} ({(1-a_winrate)*100:.1f}%)")
    print(f"Avg moves/game: {avg_moves:.1f}")
    print(f"Avg score margin for Black: {avg_margin_for_black:+.2f}")
    print(f"Avg score margin for {a.name}: {avg_margin_for_a:+.2f}")
    print(f"{a.name} avg move latency: {statistics.mean(a_move_latencies):.2f} ms")
    print(f"{b.name} avg move latency: {statistics.mean(b_move_latencies):.2f} ms")

    score_compared = [r for r in results if r.score_match is not None]
    if score_compared:
        agreed = sum(1 for r in score_compared if r.score_match)
        disagreed = len(score_compared) - agreed
        print(
            f"GNU score cross-check: {agreed}/{len(score_compared)} agree "
            f"({disagreed} disagree)"
        )
        if disagreed:
            print("  disagreements (game_idx | ours -> gnu):")
            for i, r in enumerate(results):
                if r.score_match is False:
                    print(
                        f"    {i}: {r.winner_name}({r.margin_for_black:+.1f}) "
                        f"-> {r.gnu_winner_name}({r.gnu_margin_for_black:+.1f})"
                    )

    if csv_path is not None:
        print(f"Per-game rows appended to {csv_path}")
    if md_path is not None:
        print(f"Per-game boards written to {md_path}")

    if args.gate_threshold is not None:
        passed = a_winrate >= args.gate_threshold
        print(f"Gate threshold {args.gate_threshold:.2f}: {'PASS' if passed else 'FAIL'}")
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
