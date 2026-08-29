from game.deck import Hand

class Player:
    def __init__(self, id):
        self.id = id
        
        self.first_turn_played = False
        
    @property
    def points(self):
        # Calculate the points based on the revealed cards in the hand
        if hasattr(self, 'hand'):
                
            # Sum points
            self._points = sum(
                card.value
                for row in self.hand.grid
                for card in row
                if card is not None and card.state == "revealed"
            )
        else:
            raise AttributeError("Points are None. Maybe Player does not have a hand assigned yet.")
        return self._points
        
    # Removing a column logic
    def check_if_column_revealed(self, col):
        # Check if a column is fully revealed
        return all(self.hand.grid[row][col].state == "revealed" for row in range(3))

    def check_if_column_same_value(self, col):
        # Check if a column has the same value
        return all(
            self.hand.grid[row][col].value == self.hand.grid[0][col].value
            for row in range(3)
        )

    def remove_column_if_possible(self, col):
        if self.check_if_column_revealed(col) and self.check_if_column_same_value(col):
            # Remove the column from the hand
            for row in range(3):
                print("Removing card from hand:", self.hand.grid[row][col].value)
                self.hand.grid[row][col] = (
                    None  # NOTE: For now the cards are dropped as they do not influence the game.
                )
    # ------------------------------Functions meant to be called from outside the class -------------------------------------------
    # Helper functions
    def give_hand(self, deck):
        self.hand = Hand(deck)
        
    def all_cards_revealed(self):
        # Check if all cards in the hand are revealed
        return all(
            card is None or card.state == "revealed"
            for row in self.hand.grid
            for card in row
        )
    
    def show_hand(self):
            self.hand.plot_grid_with_states()
            
    def reveal_first_two_cards(self, pos_1, pos_2):
        self.hand.reveal_first_two_cards(pos_1, pos_2)
        
    # Possible ways to play a turn ---------------------------------------
    def get_last_revealed_card_change_with_hand(self, previously_sampled_card, position):
        # Previously_sampled_card is the card that was revealed in the deck.
        # Position is the position of the card in the hand that you want to change with the revealed card in the deck.
        # Stealing the last revealed card in the deck and changing it with one of your own
        new_card = previously_sampled_card
        existing_card = self.replace_card(position, new_card)
       
        existing_card.reveal()
        # Remove a column if it is fully revealed and has the same value
        for col in range(3):
            self.remove_column_if_possible(col)
    
        return existing_card # Return the card that was replaced and reveal it
    
    def steal_card_from_deck_change_with_hand(self, new_card, position):
        # New_card is the card that was revealed in the deck.
        # Position is the position of the card in the hand that you want to change with the revealed card in the deck.

        existing_card = self.replace_card(position, new_card)
        existing_card.reveal()
        
        # Remove a column if it is fully revealed and has the same value
        for col in range(3):
            self.remove_column_if_possible(col) 
        
        return  existing_card  # Return the card that was replaced and reveal it
    
    def steal_card_from_deck_discard_reveal_from_hand(self, new_card, position):
        # Pass the new_card argument to force the deck class to reveal a card.
        assert new_card is not None, "New card must be provided to reveal a card from the deck."
        self.reveal_card(position)

    def reveal_all_cards(self):
        # Reveal all cards in the hand
        for row in range(3):
            for col in range(4):
                if self.hand.grid[row][col] is not None:
                    self.reveal_card((row, col))
    
    def get_lowest_revealed_card(self):
        # Get the lowest revealed card in the hand
        revealed_cards = [
            card
            for row in self.hand.grid
            for card in row
            if card is not None and card.state == "revealed"
        ]
        if not revealed_cards:
            raise ValueError("No revealed cards in hand.")
        return min(revealed_cards, key=lambda card: card.value)
        
    # Card Movements --------------------------------------------
    def first_turn(self, pos):
        # Call players to reveal their first two cards. 
        assert isinstance(pos, tuple), "Position must be a tuple"
        self.reveal_first_two_cards(pos_1=pos[0], pos_2=pos[1])
        self.first_turn_played = True
    
    def reveal_card(self, pos):
        self.hand.reveal_card(pos)
        
    def replace_card(self, pos, new_card):
        # Replace a card in the hand with a new card
        if len(pos) != 2:
            raise ValueError("Position must be a tuple of length 2")
        existing_card = self.hand.grid[pos[0]][pos[1]]
        
        self.hand.grid[pos[0]][pos[1]] = new_card
        new_card.give_position(pos)
        if existing_card.state == "revealed":
            new_card.reveal()
        return existing_card  # Return the card that was replaced
    

        
    
    
    
