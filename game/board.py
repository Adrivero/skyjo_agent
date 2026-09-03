# Local imports
from game.deck import Deck
from game.player import Player

# External imports
import random
from rich.console import Console
from rich.table import Table


# Contains player objects with their own cards. Keeps track of a players cards and their neighbors.
class Board:
    def __init__(self, n_players):
        self.n_players = n_players
        self.random_start_player = True

    def reset_game(self, player_ids=None, shuffle_players=True):
        self.i_turn = 0
        self.deck = Deck()
        if player_ids is None:
            player_ids = [chr(65 + i) for i in range(self.n_players)]

        self.players = [Player(player_id) for player_id in player_ids]
        for player in self.players:
            player.give_hand(self.deck)

        self.deck.start_discard_pile()

        if shuffle_players:
            self.random_player_order()

    def play_first_turn(self, initial_positions=None):
        initial_positions = initial_positions or {}
        for player in self.players:
            selected_positions = initial_positions.get(player.id)
            if selected_positions is None:
                positions = random.sample(range(12), 2)
                pos_1 = divmod(positions[0], 4)
                pos_2 = divmod(positions[1], 4)
            else:
                if len(selected_positions) != 2:
                    raise ValueError("Exactly two initial positions are required.")
                pos_1, pos_2 = (tuple(position) for position in selected_positions)
                valid_positions = {
                    (row, col) for row in range(3) for col in range(4)
                }
                if pos_1 == pos_2 or pos_1 not in valid_positions or pos_2 not in valid_positions:
                    raise ValueError("Initial positions must be distinct cells in the hand.")
            player.first_turn((pos_1, pos_2))

        self.i_turn += 1

    def play_player_action(self, player, action_type, position, drawn_card=None):
        new_deck_card = drawn_card

        if action_type == "replace":
            if new_deck_card is None:
                new_deck_card = self.deck.sample_card()
            existing_card = player.replace_card(position, new_deck_card)
            new_deck_card.reveal()
            self.deck.keep_discarded_card(existing_card)
        elif action_type == "discard_reveal":
            if new_deck_card is None:
                new_deck_card = self.deck.sample_card()
            player.steal_card_from_deck_discard_reveal_from_hand(
                new_card=new_deck_card,
                position=position,
            )
            self.deck.keep_discarded_card(new_deck_card)
        elif action_type == "take_discard_replace":
            discard_card = self.deck.take_top_discarded_card()
            existing_card = player.replace_card(position, discard_card)
            discard_card.reveal()
            self.deck.keep_discarded_card(existing_card)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        player.remove_completed_columns()

    def final_scores(self, closing_player_id=None):
        for player in self.players:
            player.reveal_all_cards()

        scoreboard = self.show_points_board()

        if closing_player_id is not None:
            if closing_player_id not in scoreboard:
                raise ValueError(f"Unknown closing player: {closing_player_id}")
            opponent_scores = [
                score
                for player_id, score in scoreboard.items()
                if player_id != closing_player_id
            ]
            if opponent_scores and scoreboard[closing_player_id] > min(opponent_scores):
                scoreboard[closing_player_id] *= 2

        return scoreboard


    def play_one_game(self,verbose = False):
        self.reset_game(shuffle_players=self.random_start_player)
        
        # First turn
        if self.i_turn == 0:
            self.play_first_turn()
            if verbose:
                # Printing
                print("First turn completed" )
                self.show_board()
                print("\n")
        
        # Play the game until a player has revealed all their cards
        if self.i_turn > 0:
            game_over = False

            while not game_over:
                for player in self.players:
                    self.heuristic_strategy(player)

                    if player.all_cards_revealed():
                        closing_player_points = player.points
                        closing_player_id = player.id
                        if verbose:
                            print(f"Player {player.id} has revealed all their cards and has {closing_player_points} points.")
                        
                        for other_player in self.players:
                            if other_player is not player:
                                self.heuristic_strategy(other_player)
                                other_player.reveal_all_cards()

                        player.reveal_all_cards()
                        game_over = True
                        break
                        
                self.i_turn += 1
                if verbose:
                    # Printing
                    print(f"Turn {self.i_turn} completed")
                    self.show_board()
                    print("\n")
        
        if verbose:
            print("Preliminary game over. Calculating points...")
            
        scoreboard = self.final_scores(closing_player_id)

        return scoreboard
        
        
    
    # Heuristic strategy:
    def heuristic_strategy(self, player):
        # Different ways to play a turn. Agent has to choose.
        # -> Heuristic: Always reveal a card from the deck. 
        # -> If lower than some revealed card in the hand, replace it. Otherwise discard it.
        # -> If the card is discarded, reveal a card from the hand at random.
        new_deck_card = self.deck.sample_card()
        revealed_cards = [
            card
            for row in player.hand.grid
            for card in row
            if card is not None and card.state == "revealed"
        ]
        lowest_revealed_card = (
            min(revealed_cards, key=lambda card: card.value)
            if revealed_cards
            else None
        )
        if (
            lowest_revealed_card is not None
            and new_deck_card.value < lowest_revealed_card.value
        ):
            self.play_player_action(
                player,
                action_type="replace",
                position=lowest_revealed_card.position,
                drawn_card=new_deck_card,
            )
        else:
            hidden_positions = [
                card.position
                for row in player.hand.grid
                for card in row
                if card is not None and card.state == "hidden"
            ]
            if hidden_positions:
                self.play_player_action(
                    player,
                    action_type="discard_reveal",
                    position=random.choice(hidden_positions),
                    drawn_card=new_deck_card,
                )
            else:
                if lowest_revealed_card is None:
                    self.deck.keep_discarded_card(new_deck_card)
                    return
                self.play_player_action(
                    player,
                    action_type="replace",
                    position=lowest_revealed_card.position,
                    drawn_card=new_deck_card,
                )


    
        
    
    # ---------------------------------------- 
    # Helper functions
    def random_player_order(self):
        random.shuffle(self.players)
        self.random_start_player = False
            
    def show_board(self):
        for player in self.players:
            print(f"Player {player.id}:")
            player.show_hand()

    def show_points_board(self,verbose = False):
        table = Table(title="Points")
        table.add_column("Player", justify="center", style="bold")
        table.add_column("Points", justify="center")

        players = {player.id: player.points for player in self.players}
        for player, points in players.items():
            table.add_row(str(player), str(points))
        if verbose:
            Console().print(table)
        return players
