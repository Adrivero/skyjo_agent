from rich.console import Console
from rich.table import Table

import random
from collections import Counter

import numpy as np

class Card:
    def __init__(self, state = "hidden", value=None):
        self.value = value # Numeric value of the card
        self.state = state # State of the card (Hidden or revealed)
        self.position = None # Position of the card in the hand (row, col)
    
    def reveal(self):
        self.state = "revealed"
        
        
    def hide(self):
        self.state = "hidden"
        
    def give_position(self, pos):
        self.position = pos
        
        


class Deck:
    # CARD_COUNTS = {
    #     -2: 5,
    #     -1: 10,
    #     0: 15,
    #     **{i: 10 for i in range(1, 13)},
    # }
    

    def __init__(self):
        self.hands = []
        self.deck = [-2] * 5 + [-1] * 10 + [0] * 15 + [i for i in range(1, 13) for _ in range(10)]
        self.heap = []
     
    def sample_card(self):
        # Sample a card and remove it from the deck.
        self.reshuffle_deck() # If the deck is empty, reshuffle the heap into the deck
        if len(self.deck) == 0:
            raise ValueError("No more cards in the deck")
        card_value = random.choice(self.deck)
        self.deck.remove(card_value)
        return Card(value=card_value)
    
    def keep_discarded_card(self, card):
        # Keep the discarded card in the deck.
        card.reveal()
        self.heap.append(card.value)

    def start_discard_pile(self):
        self.keep_discarded_card(self.sample_card())

    def top_discard_value(self):
        if not self.heap:
            return None
        return self.heap[-1]

    def take_top_discarded_card(self):
        if not self.heap:
            raise ValueError("No discarded cards available")
        return Card(state="revealed", value=self.heap.pop())
    
    # If the deck is empty, the heap is shuffled and becomes the new deck
    def reshuffle_deck(self):
        if len(self.deck) == 0:
            self.deck = self.heap
            self.heap = []
            random.shuffle(self.deck)
    
        
    


class Hand:
    def __init__(self, deck):
        self.deck = deck
        self.n_initial_cards = 12 # Number of cards in the beggining
        self.n_cards = self.n_initial_cards # Number of cards in the hand

        self.cards = []
        for _ in range(self.n_cards):
            self.cards.append(self.deck.sample_card())
        
        self.create_initial_grid()
            

    def create_initial_grid(self):
        # Create grid and assign positions to the cards in the hand
        self.grid = np.array(self.cards, dtype=object).reshape(3, 4)
        for i in range(3):
            for j in range(4):
                self.grid[i][j].give_position((i, j))
                
        # self.grid = np.array(
        #     self.cards[:self.n_initial_cards],
        #     dtype=object
        # ).reshape(3, 4)
        
    def plot_grid(self):
        table = Table(show_header=False, box=None)

        for _ in range(4):
            table.add_column(justify="center")

        for row in self.grid:
            table.add_row(
                *[str(card.value) for card in row]
            )

        Console().print(table)


    def plot_grid_with_states(self):
        table = Table(show_header=False, box=None)

        for _ in range(4):
            table.add_column(justify="center")

        for row in self.grid:
            values = []

            for card in row:
                if card is None:
                    values.append("")
                elif card.state == "hidden":
                    values.append("[bold red]🟥[/bold red]")
                else:
                    values.append(f"[bold green]{card.value}[/bold green]")

            table.add_row(*values)

        Console().print(table)   
          
    # Movement of cards in the grid
    def reveal_first_two_cards(self, pos_1, pos_2):
        # Reveal the first two cards in the grid
        if len(pos_1) != 2 or len(pos_2) != 2:
            raise ValueError("Positions must be a tuple of length 2")
        self.grid[pos_1[0]][pos_1[1]].state = "revealed"
        self.grid[pos_2[0]][pos_2[1]].state = "revealed"
    
    def reveal_card(self, pos):
        # Reveal a card in the grid
        if len(pos) != 2:
            raise ValueError("Position must be a tuple of length 2")
        self.grid[pos[0]][pos[1]].state = "revealed"

    # Plotting code 
    
        
    

        

        
        
    
