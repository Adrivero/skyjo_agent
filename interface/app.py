import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from agent.environment import DRAW_ACTION, N_CARDS, TAKE_DISCARD_OFFSET
from interface.policy import OpponentPolicy
from interface.session import SkyjoSession


class SkyjoApp:
    CARD_WIDTH = 7

    def __init__(self, root, policy_path):
        self.root = root
        self.root.title("Skyjo — Human vs Policy")
        self.root.minsize(820, 760)
        self._configure_fonts()

        self.policy = OpponentPolicy(policy_path)
        self.session = SkyjoSession(self.policy)
        self.loading_results = queue.Queue()
        self.screen = "loading"
        self.opening_positions = set()
        self.selected_move = None
        self.policy_job = None

        self.warning_text = tk.StringVar()
        self.match_text = tk.StringVar()
        self.total_text = tk.StringVar()
        self.last_match_text = tk.StringVar()
        self.status_text = tk.StringVar(value="Loading learned policy…")
        self.drawn_text = tk.StringVar(value="Drawn card: —")

        self._build_layout()
        self._refresh()
        threading.Thread(target=self._load_policy, daemon=True).start()
        self.root.after(100, self._poll_policy_load)

    def _configure_fonts(self):
        tkfont.nametofont("TkDefaultFont").configure(size=13)
        tkfont.nametofont("TkTextFont").configure(size=13)
        tkfont.nametofont("TkMenuFont").configure(size=13)

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(
            container,
            text="SKYJO",
            font=("TkDefaultFont", 28, "bold"),
        ).pack()

        self.warning_label = tk.Label(
            container,
            textvariable=self.warning_text,
            background="#fff3cd",
            foreground="#664d03",
            padx=10,
            pady=6,
            wraplength=650,
            font=("TkDefaultFont", 12, "bold"),
        )

        self.score_frame = ttk.Frame(container, padding=(0, 10))
        self.score_frame.pack(fill="x")
        ttk.Label(self.score_frame, textvariable=self.match_text).pack(side="left")
        ttk.Label(
            self.score_frame,
            textvariable=self.total_text,
            font=("TkDefaultFont", 14, "bold"),
        ).pack(side="right")
        ttk.Label(container, textvariable=self.last_match_text).pack(anchor="w")

        opponent_frame = ttk.LabelFrame(container, text="Opponent", padding=10)
        opponent_frame.pack(pady=(12, 8))
        self.opponent_cards = []
        for index in range(N_CARDS):
            label = tk.Label(
                opponent_frame,
                text="■",
                width=self.CARD_WIDTH,
                height=2,
                relief="ridge",
                background="#334155",
                foreground="#ffffff",
                font=("TkDefaultFont", 18, "bold"),
            )
            row, col = divmod(index, 4)
            label.grid(row=row, column=col, padx=4, pady=4)
            self.opponent_cards.append(label)

        table_frame = ttk.Frame(container, padding=8)
        table_frame.pack()
        self.draw_button = ttk.Button(
            table_frame,
            text="Draw",
            command=self._draw_card,
        )
        self.draw_button.grid(row=0, column=0, padx=8)
        self.discard_button = ttk.Button(
            table_frame,
            text="Pile: —",
            command=self._choose_discard,
        )
        self.discard_button.grid(row=0, column=1, padx=8)
        ttk.Label(table_frame, textvariable=self.drawn_text).grid(
            row=0, column=2, padx=14
        )

        action_frame = ttk.Frame(container)
        action_frame.pack(pady=(0, 8))
        self.replace_button = ttk.Button(
            action_frame,
            text="Replace a card",
            command=lambda: self._select_move("replace"),
        )
        self.replace_button.pack(side="left", padx=5)
        self.reveal_button = ttk.Button(
            action_frame,
            text="Discard draw + reveal",
            command=lambda: self._select_move("reveal"),
        )
        self.reveal_button.pack(side="left", padx=5)

        human_frame = ttk.LabelFrame(container, text="You", padding=10)
        human_frame.pack(pady=8)
        self.human_cards = []
        for index in range(N_CARDS):
            button = tk.Button(
                human_frame,
                text="■",
                width=self.CARD_WIDTH,
                height=2,
                command=lambda card_index=index: self._click_human_card(card_index),
                disabledforeground="#2f343a",
                font=("TkDefaultFont", 18, "bold"),
                relief="raised",
            )
            row, col = divmod(index, 4)
            button.grid(row=row, column=col, padx=4, pady=4)
            self.human_cards.append(button)

        self.status_label = tk.Label(
            container,
            textvariable=self.status_text,
            pady=10,
            wraplength=650,
            font=("TkDefaultFont", 14, "bold"),
        )
        self.status_label.pack()

        footer = ttk.Frame(container)
        footer.pack(pady=4)
        self.start_button = ttk.Button(
            footer,
            text="Start match",
            command=self._start_match,
        )
        self.start_button.pack(side="left", padx=5)
        self.next_button = ttk.Button(
            footer,
            text="Next match",
            command=self._prepare_opening,
        )
        self.next_button.pack(side="left", padx=5)
        self.new_series_button = ttk.Button(
            footer,
            text="New series",
            command=self._new_series,
        )
        self.new_series_button.pack(side="left", padx=5)

    def _load_policy(self):
        loaded = self.policy.load()
        self.loading_results.put(loaded)

    def _poll_policy_load(self):
        try:
            self.loading_results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_policy_load)
            return
        self._prepare_opening()

    def _prepare_opening(self):
        self.screen = "opening"
        self.opening_positions.clear()
        self.selected_move = None
        self._refresh()

    def _start_match(self):
        if len(self.opening_positions) != 2:
            return
        positions = [divmod(index, 4) for index in sorted(self.opening_positions)]
        self.session.start_match(positions)
        self.screen = "playing"
        self.selected_move = None
        self._refresh()
        self._schedule_policy()

    def _new_series(self):
        if self.policy_job is not None:
            self.root.after_cancel(self.policy_job)
            self.policy_job = None
        self.session.new_series()
        self._prepare_opening()

    def _draw_card(self):
        self.session.human_draw()
        self.selected_move = None
        self._refresh()

    def _choose_discard(self):
        self.selected_move = "take"
        self._refresh()

    def _select_move(self, move):
        self.selected_move = move
        self._refresh()

    def _click_human_card(self, index):
        if self.screen == "opening":
            if index in self.opening_positions:
                self.opening_positions.remove(index)
            elif len(self.opening_positions) < 2:
                self.opening_positions.add(index)
            self._refresh()
            return

        if not self.session.human_turn:
            return
        if self.selected_move == "take":
            self.session.human_take_discard(index)
        elif self.selected_move == "replace":
            self.session.human_replace_with_drawn(index)
        elif self.selected_move == "reveal":
            self.session.human_discard_and_reveal(index)
        else:
            return

        self.selected_move = None
        self._refresh()
        self._schedule_policy()

    def _schedule_policy(self):
        if (
            self.policy_job is None
            and self.session.match_active
            and self.session.current_agent == self.session.policy_agent
        ):
            self.policy_job = self.root.after(450, self._step_policy)

    def _step_policy(self):
        self.policy_job = None
        try:
            self.session.step_policy()
        except Exception as error:
            self.warning_text.set(f"Opponent turn failed: {type(error).__name__}: {error}")
            self.status_text.set("The match cannot continue.")
            return
        self._refresh()
        self._schedule_policy()

    def _refresh(self):
        self._refresh_warning()
        self._refresh_scores()
        self._refresh_cards()
        self._refresh_controls()
        self._refresh_status()

    def _refresh_warning(self):
        warning = self.policy.warning or ""
        self.warning_text.set(warning)
        if warning:
            self.warning_label.pack(
                fill="x", pady=(8, 0), before=self.score_frame
            )
        else:
            self.warning_label.pack_forget()

    def _refresh_scores(self):
        match_number = self.session.match_number or 1
        if self.screen == "opening":
            match_number = len(self.session.match_history) + 1
        self.match_text.set(f"Match {match_number} · First to 100 ends the series")
        self.total_text.set(
            f"Totals — You: {self.session.totals[self.session.human_agent]}   "
            f"Opponent: {self.session.totals[self.session.policy_agent]}"
        )
        last_match = self.session.last_match
        if last_match is None:
            self.last_match_text.set("Previous match: —")
        else:
            scores = last_match["scores"]
            self.last_match_text.set(
                f"Previous match — You: {scores[self.session.human_agent]}   "
                f"Opponent: {scores[self.session.policy_agent]}"
            )

    def _refresh_cards(self):
        if self.session.environment is None or self.screen == "opening":
            for index, button in enumerate(self.human_cards):
                selected = index in self.opening_positions
                button.configure(
                    text="✓" if selected else "■",
                    background="#38bdf8" if selected else "#334155",
                    foreground="#082f49" if selected else "#ffffff",
                    disabledforeground="#082f49" if selected else "#ffffff",
                    relief="sunken" if selected else "raised",
                )
            for label in self.opponent_cards:
                label.configure(
                    text="■", background="#334155", foreground="#ffffff"
                )
            self.drawn_text.set("Drawn card: —")
            self.discard_button.configure(text="Pile: —")
            return

        self._render_hand(self.session.policy_agent, self.opponent_cards)
        self._render_hand(self.session.human_agent, self.human_cards)
        deck = self.session.environment.board.deck
        self.discard_button.configure(text=f"Pile: {deck.top_discard_value()}")
        pending = self.session.environment.pending_drawn_card
        self.drawn_text.set(
            f"Drawn card: {pending.value}" if pending is not None else "Drawn card: —"
        )

    def _render_hand(self, agent, widgets):
        player = self.session.player(agent)
        for index, card in enumerate(player.hand.grid.flatten()):
            if card is None:
                text, background, foreground = "—", "#e2e8f0", "#475569"
            elif card.state == "hidden":
                text, background, foreground = "■", "#334155", "#ffffff"
            else:
                text = str(card.value)
                background, foreground = self._card_colors(card.value)
            widgets[index].configure(
                text=text,
                background=background,
                foreground=foreground,
                relief="ridge",
            )
            if isinstance(widgets[index], tk.Button):
                widgets[index].configure(disabledforeground=foreground)

    def _refresh_controls(self):
        for widget in (
            self.draw_button,
            self.discard_button,
            self.replace_button,
            self.reveal_button,
            self.start_button,
            self.next_button,
            self.new_series_button,
        ):
            widget.configure(state="disabled")
        for button in self.human_cards:
            button.configure(state="disabled")

        if self.screen == "loading":
            return
        if self.screen == "opening":
            for button in self.human_cards:
                button.configure(state="normal")
            if len(self.opening_positions) == 2:
                self.start_button.configure(state="normal")
            if self.session.match_history:
                self.new_series_button.configure(state="normal")
            return
        if self.session.series_over:
            self.new_series_button.configure(state="normal")
            return
        if self.session.match_over:
            self.next_button.configure(state="normal")
            self.new_series_button.configure(state="normal")
            return
        if not self.session.human_turn:
            return

        observation = self.session.environment.observe(self.session.human_agent)
        mask = observation["action_mask"]
        if self.session.turn_phase == "choose_source":
            if mask[DRAW_ACTION]:
                self.draw_button.configure(state="normal")
            if any(mask[TAKE_DISCARD_OFFSET:]):
                self.discard_button.configure(state="normal")
            if self.selected_move == "take":
                for index, button in enumerate(self.human_cards):
                    if mask[TAKE_DISCARD_OFFSET + index]:
                        button.configure(state="normal")
        else:
            self.replace_button.configure(state="normal")
            if any(mask[N_CARDS : N_CARDS * 2]):
                self.reveal_button.configure(state="normal")
            if self.selected_move == "replace":
                for index, button in enumerate(self.human_cards):
                    if mask[index]:
                        button.configure(state="normal")
            elif self.selected_move == "reveal":
                for index, button in enumerate(self.human_cards):
                    if mask[N_CARDS + index]:
                        button.configure(state="normal")

    def _refresh_status(self):
        if self.screen == "loading":
            self.status_text.set("Loading and validating the learned policy…")
            return
        if self.screen == "opening":
            remaining = 2 - len(self.opening_positions)
            self.status_text.set(
                "Choose two cards to reveal, then start the match."
                if remaining
                else "Opening cards selected. Start the match when ready."
            )
            return
        if self.session.series_over:
            winner = self.session.series_winner
            if winner == "tie":
                result = "The series ends in a tie."
            elif winner == self.session.human_agent:
                result = "You win the series!"
            else:
                result = "The opponent wins the series."
            self.status_text.set(f"{result} Start a new series to play again.")
            return
        if self.session.match_over:
            record = self.session.last_match
            scores = record["scores"]
            raw = record["raw_scores"]
            penalty = ""
            closer = record["closing_agent"]
            if closer is not None and scores[closer] != raw[closer]:
                name = "Your" if closer == self.session.human_agent else "Opponent's"
                penalty = f" {name} closing score doubled from {raw[closer]} to {scores[closer]}."
            self.status_text.set(
                f"Match complete — You {scores[self.session.human_agent]}, "
                f"Opponent {scores[self.session.policy_agent]}.{penalty}"
            )
            return

        closing_agent = self.session.environment.closing_agent
        final_round = " Final turn of the match." if closing_agent is not None else ""
        if not self.session.human_turn:
            self.status_text.set(f"Opponent is playing…{final_round}")
        elif self.session.turn_phase == "choose_source":
            if self.selected_move == "take":
                self.status_text.set("Choose a card to replace with the discard.")
            else:
                self.status_text.set(f"Your turn: draw a card or take the discard.{final_round}")
        elif self.selected_move == "replace":
            self.status_text.set("Choose a card to replace with the drawn card.")
        elif self.selected_move == "reveal":
            self.status_text.set("Choose a hidden card to reveal.")
        else:
            self.status_text.set("Use the drawn card, or discard it and reveal a hidden card.")

    @staticmethod
    def _card_colors(value):
        if value <= 0:
            return "#22c55e", "#052e16"
        if value <= 4:
            return "#facc15", "#422006"
        if value <= 8:
            return "#fb923c", "#431407"
        return "#ef4444", "#ffffff"


def run(policy_path):
    root = tk.Tk()
    SkyjoApp(root, policy_path)
    root.mainloop()
