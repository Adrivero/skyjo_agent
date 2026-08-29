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


    def play_one_game(self,verbose = False):
        self.i_turn = 0
        self.deck = Deck()
        self.players = [Player(chr(65 + i)) for i in range(self.n_players)]
        for player in self.players:
            player.give_hand(self.deck)
            
        if self.random_start_player:
            self.random_player_order()
        
        # First turn
        if self.i_turn == 0:
            for player in self.players:
                # Here you can implement the logic for the player to choose two cards to reveal
                # For now, we will just reveal two random cards
                pos_1 = (random.randint(0, 2), random.randint(0, 3))
                pos_2 = (random.randint(0, 2), random.randint(0, 3))
                player.first_turn((pos_1, pos_2))
                
            self.i_turn += 1
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
            
        scoreboard = self.show_points_board()
        
        # Determine if the winner with the lowest points was the player who closed the game or if another player has lower points.
        least_points_player = min(scoreboard, key=scoreboard.get)

        if  least_points_player == closing_player_id:
            if verbose:
                print("The player who closed the game has the least points.")
        else:
            if verbose:
                print("Another player has the least points. Multiplying closing player's points by 2.")
            scoreboard[closing_player_id] *= 2

        return scoreboard
        
        
    
    # Heuristic strategy:
    def heuristic_strategy(self, player):
        # Play normal turn
        new_deck_card = self.deck.sample_card()
        
        # Different ways to play a turn. Agent has to choose.
        # -> Heuristic: Always reveal a card from the deck. 
        # -> If lower than some revealed card in the hand, replace it. Otherwise discard it.
        # -> If the card is discarded, reveal a card from the hand at random.
        lowest_revealed_card = player.get_lowest_revealed_card()
        if new_deck_card.value < lowest_revealed_card.value:
            existing_card = player.replace_card(lowest_revealed_card.position, new_deck_card)
            # Add the existing card to the deck's discarded pile
            self.deck.keep_discarded_card(existing_card)
        else:
            random_pos = (random.randint(0, 2), random.randint(0, 3))
            player.steal_card_from_deck_discard_reveal_from_hand(new_card=new_deck_card, position=random_pos)
            self.deck.keep_discarded_card(new_deck_card)


    
        
    
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
