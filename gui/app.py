"""
Pygame GUI for 9x9 Go.

State machine
-------------
START_SCREEN -> PLAYER_TURN <-> AI_THINKING -> GAME_OVER -> START_SCREEN

Coordinate convention
---------------------
Internal (r, c) follows the engine: r=0 is the top row, c=0 is the leftmost
column. Display labels follow Go convention: columns A B C D E F G H J
(skipping I to avoid confusion with 1), rows numbered 9 at the top down to
1 at the bottom. Labels are display-only; click handling converts pixel ->
internal (r, c).

Pass policy
-----------
Two distinct sidebar buttons:

- "Pass" calls GoGame.pass_turn(): a real pass that hands the turn to the
  AI. Two consecutive passes end the game and trigger territory scoring.
  No confirmation needed — passing is harmless under area scoring.
- "End game" calls GoGame.concede(): an immediate forfeit. The human loses
  regardless of board position. Two-click confirmation in red, since this
  is destructive.

When the AI's agent returns "pass", the engine's pass_turn() runs and the
GUI waits for the human's reply.
"""

from __future__ import annotations

import argparse
from typing import Optional

import pygame

from ai.agent import MCTSAgent
from ai.puct_agent import PUCTAgent
from engine.go_engine import BLACK, EMPTY, SIZE, WHITE, GoGame

# --- Layout ---
CELL_PX = 70
MARGIN = 70
BOARD_PX = (SIZE - 1) * CELL_PX
BOARD_RIGHT = MARGIN + BOARD_PX
SIDEBAR_X = BOARD_RIGHT + 50
SIDEBAR_W = 260
WINDOW_W = SIDEBAR_X + SIDEBAR_W
WINDOW_H = MARGIN + BOARD_PX + MARGIN

STONE_R = CELL_PX // 2 - 3
LAST_DOT_R = 5
LABEL_PAD = 28
TERRITORY_R = 7

# --- Colors ---
BG = (220, 210, 190)
BG_BOARD = (235, 195, 130)
GRID_C = (50, 35, 20)
STONE_B = (25, 25, 25)
STONE_W = (245, 245, 240)
STONE_OUTLINE = (30, 30, 30)
LAST_DOT_C = (210, 50, 50)
TEXT_C = (25, 25, 25)
TEXT_MUTED = (80, 75, 70)
PANEL_BG = (215, 200, 165)
BUTTON_BG_C = (195, 180, 145)
BUTTON_HOVER_C = (210, 195, 160)
BUTTON_ACTIVE_C = (180, 165, 130)
BUTTON_DANGER_C = (215, 130, 100)
OVERLAY_C = (0, 0, 0, 150)
OVERLAY_TEXT_C = (240, 240, 240)
TERRITORY_B = (25, 25, 25, 120)
TERRITORY_W = (245, 245, 240, 140)

START_SCREEN = "start"
PLAYER_TURN = "player"
AI_THINKING = "ai"
GAME_OVER = "over"

COL_LABELS = "ABCDEFGHJ"
STAR_POINTS = ((2, 2), (2, 6), (4, 4), (6, 2), (6, 6))
FPS = 30

ILLEGAL_MSG_MS = 3000
ILLEGAL_PREVIEW_C = (220, 50, 50, 130)
ILLEGAL_TEXT_C = (200, 30, 30)


def coord_label(r: int, c: int) -> str:
    return f"{COL_LABELS[c]}{SIZE - r}"


def intersection_xy(r: int, c: int) -> tuple[int, int]:
    return (MARGIN + c * CELL_PX, MARGIN + r * CELL_PX)


def xy_to_rc(x: int, y: int) -> Optional[tuple[int, int]]:
    """Snap pixel to nearest intersection. Reject if cursor is too far away."""
    cc = round((x - MARGIN) / CELL_PX)
    rr = round((y - MARGIN) / CELL_PX)
    if not (0 <= cc < SIZE and 0 <= rr < SIZE):
        return None
    px, py = intersection_xy(rr, cc)
    if (px - x) ** 2 + (py - y) ** 2 > (CELL_PX // 2) ** 2:
        return None
    return (rr, cc)


class Button:
    def __init__(self, rect: tuple[int, int, int, int], text: str) -> None:
        self.rect = pygame.Rect(rect)
        self.text = text

    def draw(
        self,
        surf: pygame.Surface,
        font: pygame.font.Font,
        hover: bool = False,
        active: bool = False,
        override_text: Optional[str] = None,
        override_bg: Optional[tuple[int, int, int]] = None,
    ) -> None:
        if override_bg is not None:
            bg = override_bg
        elif active:
            bg = BUTTON_ACTIVE_C
        elif hover:
            bg = BUTTON_HOVER_C
        else:
            bg = BUTTON_BG_C
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, GRID_C, self.rect, width=1, border_radius=6)
        ts = font.render(override_text or self.text, True, TEXT_C)
        surf.blit(ts, ts.get_rect(center=self.rect.center))

    def hit(self, pos: tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class App:
    def __init__(
        self,
        default_human_color: int,
        ai_iterations: int,
        ai_engine: str,
        model_path: Optional[str],
        seed: Optional[int],
    ) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption("9x9 Go")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.small_font = pygame.font.SysFont("arial", 14)
        self.label_font = pygame.font.SysFont("arial", 14, bold=True)
        self.title_font = pygame.font.SysFont("arial", 28, bold=True)

        self.default_human_color = default_human_color
        self.ai_iterations = ai_iterations
        self.ai_engine = ai_engine
        self.model_path = model_path
        self.seed = seed

        self.game = GoGame()
        self.agent = self._build_agent()
        self.human_color = default_human_color
        self.state = START_SCREEN
        self.score_dict: Optional[dict] = None
        self.confirm_end = False
        self.illegal_msg: Optional[str] = None
        self.illegal_msg_time = 0

        self.show_black_terr = False
        self.show_white_terr = False

        # Undo history: snapshots taken before the player's move
        self._history: list = []

        # Pre-rendered territory dots
        self.terr_b_surf = pygame.Surface((TERRITORY_R * 2, TERRITORY_R * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.terr_b_surf, TERRITORY_B, (TERRITORY_R, TERRITORY_R), TERRITORY_R)
        self.terr_w_surf = pygame.Surface((TERRITORY_R * 2, TERRITORY_R * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.terr_w_surf, TERRITORY_W, (TERRITORY_R, TERRITORY_R), TERRITORY_R)

        self.black_terr_btn = Button((SIDEBAR_X, MARGIN + 280, 105, 32), "Black territory")
        self.white_terr_btn = Button((SIDEBAR_X + 115, MARGIN + 280, 105, 32), "White territory")
        self.undo_btn = Button((SIDEBAR_X, MARGIN + 430, 220, 40), "Undo")
        self.pass_btn = Button((SIDEBAR_X, MARGIN + 480, 220, 40), "Pass")
        self.end_btn = Button((SIDEBAR_X, MARGIN + 530, 220, 40), "End game")
        self.new_btn = Button((SIDEBAR_X, MARGIN + 580, 220, 40), "New game")

        cx = WINDOW_W // 2
        cy = WINDOW_H // 2
        self.start_black_btn = Button((cx - 200, cy + 20, 180, 60), "Black (1st)")
        self.start_white_btn = Button((cx + 20, cy + 20, 180, 60), "White (2nd)")
        self.over_new_btn = Button((cx - 90, cy + 90, 180, 40), "New game")

    def _build_agent(self):
        if self.ai_engine == "puct":
            return PUCTAgent(iterations=self.ai_iterations, model_path=self.model_path, seed=self.seed)
        return MCTSAgent(iterations=self.ai_iterations, seed=self.seed)

    # --- Main loop ---

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            self._render()

            # AI runs synchronously *after* the "AI thinking..." frame
            # has been flipped, so the user sees the spinner.
            if self.state == AI_THINKING:
                self._do_ai_move()

            self.clock.tick(FPS)

        pygame.quit()

    # --- Event handling ---

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.state == START_SCREEN:
            if self.start_black_btn.hit(pos):
                self._start_game(BLACK)
            elif self.start_white_btn.hit(pos):
                self._start_game(WHITE)
            return

        if self.state == GAME_OVER:
            if self.over_new_btn.hit(pos):
                self.state = START_SCREEN
                self.confirm_end = False
            if self.black_terr_btn.hit(pos):
                self.show_black_terr = not self.show_black_terr
            if self.white_terr_btn.hit(pos):
                self.show_white_terr = not self.show_white_terr
            return

        if self.state != PLAYER_TURN:
            return

        if self.black_terr_btn.hit(pos):
            self.show_black_terr = not self.show_black_terr
            self.confirm_end = False
            return
        if self.white_terr_btn.hit(pos):
            self.show_white_terr = not self.show_white_terr
            self.confirm_end = False
            return

        if self.undo_btn.hit(pos):
            if self._history:
                snapshot = self._history.pop()
                self.game = snapshot
                self.confirm_end = False
            return

        if self.pass_btn.hit(pos):
            # Real pass: hand the turn to the AI. Two consecutive passes
            # (one from each side) end the game and trigger scoring.
            self.confirm_end = False
            self.game.pass_turn()
            if self.game.finished:
                self._end_game()
            else:
                self.state = AI_THINKING
            return

        if self.end_btn.hit(pos):
            if self.confirm_end:
                # "I give up" — concede ends the game with the human as
                # the loser regardless of board position.
                self.game.concede()
                self._end_game()
            else:
                self.confirm_end = True
            return

        if self.new_btn.hit(pos):
            self.state = START_SCREEN
            self.confirm_end = False
            return

        self.confirm_end = False  # any other click cancels confirmation
        rc = xy_to_rc(*pos)
        if rc is None:
            return
        r, c = rc
        snapshot = self.game.clone_fast()
        if self.game.place_stone(r, c):
            self.illegal_msg = None
            self._history.append(snapshot)
            if self.game.finished:
                self._end_game()
            elif self.game.to_move != self.human_color:
                self.state = AI_THINKING
        else:
            # Engine returned False: occupied, suicide, or ko. The engine
            # doesn't tell us which, so the message is generic.
            self.illegal_msg = f"Illegal: {coord_label(r, c)}"
            self.illegal_msg_time = pygame.time.get_ticks()

    def _start_game(self, human_color: int) -> None:
        self.game = GoGame()
        self.agent = self._build_agent()
        self.human_color = human_color
        self.score_dict = None
        self.confirm_end = False
        self.illegal_msg = None
        self.show_black_terr = False
        self.show_white_terr = False
        self._history.clear()
        self.state = PLAYER_TURN if self.game.to_move == self.human_color else AI_THINKING

    def _do_ai_move(self) -> None:
        move = self.agent.select_move(self.game)
        if move == "pass":
            self.game.pass_turn()
            if self.game.finished:
                # Second consecutive pass ended the game — score it.
                self._end_game()
            else:
                self.state = PLAYER_TURN
            return
        if not self.game.place_stone(*move):
            # Defensive: agent shouldn't return illegal moves, but if it
            # does, treat it as a forfeit rather than crashing.
            self.game.concede()
            self._end_game()
            return
        if self.game.finished:
            self._end_game()
        else:
            self.state = PLAYER_TURN

    def _end_game(self) -> None:
        self.score_dict = self.game.score_final()
        self.state = GAME_OVER
        self.confirm_end = False

    # --- Rendering ---

    def _render(self) -> None:
        self.screen.fill(BG)
        if self.state == START_SCREEN:
            self._render_start()
        else:
            self._render_board()
            self._render_sidebar()
            if self.state == AI_THINKING:
                self._render_overlay("AI thinking...")
            elif self.state == GAME_OVER:
                self._render_game_over()
        pygame.display.flip()

    def _render_start(self) -> None:
        cx = WINDOW_W // 2
        cy = WINDOW_H // 2
        title = self.title_font.render("9x9 Go", True, TEXT_C)
        self.screen.blit(title, title.get_rect(center=(cx, cy - 80)))
        subtitle = self.font.render("Choose your color", True, TEXT_C)
        self.screen.blit(subtitle, subtitle.get_rect(center=(cx, cy - 30)))
        note = self.small_font.render("Black moves first. Pass = concede.", True, TEXT_C)
        self.screen.blit(note, note.get_rect(center=(cx, cy)))
        mouse = pygame.mouse.get_pos()
        self.start_black_btn.draw(self.screen, self.font, hover=self.start_black_btn.hit(mouse))
        self.start_white_btn.draw(self.screen, self.font, hover=self.start_white_btn.hit(mouse))

    def _render_board(self) -> None:
        for i in range(SIZE):
            y = MARGIN + i * CELL_PX
            pygame.draw.line(self.screen, GRID_C, (MARGIN, y), (BOARD_RIGHT, y), 1)
            x = MARGIN + i * CELL_PX
            pygame.draw.line(self.screen, GRID_C, (x, MARGIN), (x, MARGIN + BOARD_PX), 1)
        for r, c in STAR_POINTS:
            pygame.draw.circle(self.screen, GRID_C, intersection_xy(r, c), 3)

        for c in range(SIZE):
            ts = self.label_font.render(COL_LABELS[c], True, TEXT_C)
            x = MARGIN + c * CELL_PX
            self.screen.blit(ts, ts.get_rect(center=(x, MARGIN - LABEL_PAD)))
            self.screen.blit(ts, ts.get_rect(center=(x, MARGIN + BOARD_PX + LABEL_PAD)))
        for r in range(SIZE):
            ts = self.label_font.render(str(SIZE - r), True, TEXT_C)
            y = MARGIN + r * CELL_PX
            self.screen.blit(ts, ts.get_rect(center=(MARGIN - LABEL_PAD, y)))
            self.screen.blit(ts, ts.get_rect(center=(MARGIN + BOARD_PX + LABEL_PAD, y)))

        for r in range(SIZE):
            for c in range(SIZE):
                v = self.game.board[r * SIZE + c]
                if v == EMPTY:
                    continue
                color = STONE_B if v == BLACK else STONE_W
                pos = intersection_xy(r, c)
                pygame.draw.circle(self.screen, color, pos, STONE_R)
                pygame.draw.circle(self.screen, STONE_OUTLINE, pos, STONE_R, 1)

        lm = self.game.last_move
        if isinstance(lm, tuple):
            pygame.draw.circle(self.screen, LAST_DOT_C, intersection_xy(*lm), LAST_DOT_R)

        # Territory dots
        if self.show_black_terr or self.show_white_terr:
            owner = self.game.ownership_map()
            for r in range(SIZE):
                for c in range(SIZE):
                    idx = r * SIZE + c
                    if self.game.board[idx] != EMPTY:
                        continue
                    o = owner[idx]
                    if o > 0 and self.show_black_terr:
                        px, py = intersection_xy(r, c)
                        self.screen.blit(self.terr_b_surf, (px - TERRITORY_R, py - TERRITORY_R))
                    elif o < 0 and self.show_white_terr:
                        px, py = intersection_xy(r, c)
                        self.screen.blit(self.terr_w_surf, (px - TERRITORY_R, py - TERRITORY_R))

        if self.state == PLAYER_TURN:
            mouse = pygame.mouse.get_pos()
            rc = xy_to_rc(*mouse)
            if rc is not None and self.game.board[rc[0] * SIZE + rc[1]] == EMPTY:
                if self.game.is_legal(*rc):
                    color = (*STONE_B, 110) if self.human_color == BLACK else (*STONE_W, 140)
                else:
                    color = ILLEGAL_PREVIEW_C
                preview = pygame.Surface((STONE_R * 2, STONE_R * 2), pygame.SRCALPHA)
                pygame.draw.circle(preview, color, (STONE_R, STONE_R), STONE_R)
                px, py = intersection_xy(*rc)
                self.screen.blit(preview, (px - STONE_R, py - STONE_R))

    def _render_sidebar(self) -> None:
        rect = pygame.Rect(SIDEBAR_X - 20, MARGIN - 30, SIDEBAR_W, BOARD_PX + 120)
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, GRID_C, rect, width=1, border_radius=8)

        x = SIDEBAR_X
        y = MARGIN - 10

        # --- Players ---
        you_color = "Black" if self.human_color == BLACK else "White"
        ai_color = "White" if self.human_color == BLACK else "Black"
        ht = self.small_font.render(f"You: {you_color}", True, TEXT_C)
        at = self.small_font.render(
            f"AI: {ai_color} ({self.ai_engine.upper()} {self.ai_iterations})", True, TEXT_MUTED,
        )
        self.screen.blit(ht, (x, y))
        self.screen.blit(at, (x, y + 18))
        y += 45

        # --- Turn ---
        label = self.font.render("To move:", True, TEXT_C)
        self.screen.blit(label, (x, y))
        dot_color = STONE_B if self.game.to_move == BLACK else STONE_W
        pygame.draw.circle(self.screen, dot_color, (x + 100, y + 10), 10)
        pygame.draw.circle(self.screen, STONE_OUTLINE, (x + 100, y + 10), 10, 1)
        who = "you" if self.game.to_move == self.human_color else "AI"
        name = self.small_font.render(f"({who})", True, TEXT_MUTED)
        self.screen.blit(name, (x + 120, y + 4))
        y += 40

        # --- Captures ---
        cb = self.font.render(f"Black captures: {self.game.captures[BLACK]}", True, TEXT_C)
        cw = self.font.render(f"White captures: {self.game.captures[WHITE]}", True, TEXT_C)
        self.screen.blit(cb, (x, y))
        self.screen.blit(cw, (x, y + 25))
        y += 60

        # --- Score (exact final, with dead-stone removal) ---
        sc = self.game.score_final()
        sh = self.font.render("Score:", True, TEXT_C)
        self.screen.blit(sh, (x, y))
        bs = self.small_font.render(
            f"Black: {sc['black']:.1f}  ({sc['black_stones']}+{sc['black_territory']})",
            True, TEXT_C,
        )
        ws = self.small_font.render(
            f"White: {sc['white']:.1f}  ({sc['white_stones']}+{sc['white_territory']}+{sc['komi']})",
            True, TEXT_C,
        )
        self.screen.blit(bs, (x, y + 22))
        self.screen.blit(ws, (x, y + 40))
        y += 70

        # --- Last Moves ---
        self.screen.blit(self.small_font.render("Last moves:", True, TEXT_MUTED), (x, y))
        y += 18
        hist = self.game.history
        if hist:
            start = max(0, len(hist) - 4)
            for i, mv in enumerate(hist[start:], start=start):
                color = "B" if i % 2 == 0 else "W"
                txt = f"{color}  {coord_label(*mv)}" if isinstance(mv, tuple) else f"{color}  {mv}"
                self.screen.blit(self.small_font.render(txt, True, TEXT_C), (x + 8, y))
                y += 18
        else:
            self.screen.blit(self.small_font.render("—", True, TEXT_MUTED), (x + 8, y))
            y += 18
        y += 10

        # --- Territory toggles ---
        mouse = pygame.mouse.get_pos()
        b_bg = BUTTON_ACTIVE_C if self.show_black_terr else BUTTON_BG_C
        w_bg = BUTTON_ACTIVE_C if self.show_white_terr else BUTTON_BG_C
        self.black_terr_btn.draw(
            self.screen, self.small_font,
            hover=self.black_terr_btn.hit(mouse), active=self.show_black_terr,
            override_bg=b_bg,
        )
        self.white_terr_btn.draw(
            self.screen, self.small_font,
            hover=self.white_terr_btn.hit(mouse), active=self.show_white_terr,
            override_bg=w_bg,
        )
        y += 42

        # --- Action buttons ---
        if self.state == PLAYER_TURN:
            self.undo_btn.draw(
                self.screen, self.font,
                hover=self.undo_btn.hit(mouse),
                active=len(self._history) > 0,
            )
            self.pass_btn.draw(self.screen, self.font, hover=self.pass_btn.hit(mouse))
            if self.confirm_end:
                self.end_btn.draw(
                    self.screen, self.small_font,
                    hover=self.end_btn.hit(mouse),
                    override_text="Confirm: I give up?",
                    override_bg=BUTTON_DANGER_C,
                )
            else:
                self.end_btn.draw(self.screen, self.font, hover=self.end_btn.hit(mouse))
            self.new_btn.draw(self.screen, self.font, hover=self.new_btn.hit(mouse))

        # Hover coordinate
        if self.state == PLAYER_TURN:
            rc = xy_to_rc(*mouse)
            if rc is not None:
                ct = self.small_font.render(f"Hover: {coord_label(*rc)}", True, TEXT_MUTED)
                self.screen.blit(ct, (x, y + 55))

        # Illegal move toast
        if (
            self.illegal_msg is not None
            and pygame.time.get_ticks() - self.illegal_msg_time < ILLEGAL_MSG_MS
        ):
            im = self.font.render(self.illegal_msg, True, ILLEGAL_TEXT_C)
            self.screen.blit(im, (x, y + 80))

    def _render_overlay(self, text: str) -> None:
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill(OVERLAY_C)
        self.screen.blit(overlay, (0, 0))
        ts = self.title_font.render(text, True, OVERLAY_TEXT_C)
        self.screen.blit(ts, ts.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2)))

    def _render_game_over(self) -> None:
        if self.score_dict is None:
            return
        s = self.score_dict
        winner_name = "Black" if s["winner"] == BLACK else "White"
        you_won = s["winner"] == self.human_color
        outcome = "You win!" if you_won else "AI wins"

        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill(OVERLAY_C)
        self.screen.blit(overlay, (0, 0))

        cx = WINDOW_W // 2
        cy = WINDOW_H // 2

        ts = self.title_font.render(f"{winner_name} wins", True, OVERLAY_TEXT_C)
        self.screen.blit(ts, ts.get_rect(center=(cx, cy - 90)))
        sub = self.font.render(outcome, True, OVERLAY_TEXT_C)
        self.screen.blit(sub, sub.get_rect(center=(cx, cy - 50)))

        lines = [
            f"Black: {s['black_stones']} stones + {s['black_territory']} territory = {s['black']:.1f}",
            f"White: {s['white_stones']} + {s['white_territory']} + {s['komi']} komi = {s['white']:.1f}",
            f"Neutral (dame): {s['neutral_points']}",
        ]
        for i, line in enumerate(lines):
            t = self.small_font.render(line, True, OVERLAY_TEXT_C)
            self.screen.blit(t, t.get_rect(center=(cx, cy - 10 + i * 22)))

        mouse = pygame.mouse.get_pos()
        self.over_new_btn.draw(self.screen, self.font, hover=self.over_new_btn.hit(mouse))


def run_app(
    human_color: int = BLACK,
    ai_iterations: int = 400,
    ai_engine: str = "puct",
    model_path: Optional[str] = "model/model_v1.pt",
    seed: Optional[int] = None,
) -> None:
    App(human_color, ai_iterations, ai_engine, model_path, seed).run()


def main() -> None:
    p = argparse.ArgumentParser(description="9x9 Go vs MCTS/PUCT AI")
    p.add_argument("--human-color", choices=["black", "white"], default="black")
    p.add_argument("--ai-iterations", type=int, default=400)
    p.add_argument("--engine", choices=["mcts", "puct"], default="puct")
    p.add_argument("--model-path", type=str, default="model/model_v1.pt")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    color = BLACK if args.human_color == "black" else WHITE
    run_app(
        human_color=color,
        ai_iterations=args.ai_iterations,
        ai_engine=args.engine,
        model_path=args.model_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
