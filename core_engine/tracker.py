from typing import List, Optional
from core_engine.rules import Combo, ComboType

class GameTracker:
    def __init__(self):
        self.reset_game()
        
    def reset_game(self):
        """Reset the game state for a new match."""
        # 1 if played, 0 if not
        self.played_cards = [0] * 52
        # Number of cards remaining for each player
        self.player_card_counts = [13, 13, 13, 13]
        # True if player has passed in the current round
        self.passed_players = [False, False, False, False]
        # The combo currently dominating the table
        self.current_combo: Optional[Combo] = None
        # The player who played the current combo
        self.controlling_player = -1
        # Is this the very first turn of the game?
        self.is_first_turn = True
        
    def end_round(self):
        """Called when 3 players have passed and 1 player takes the table."""
        self.current_combo = None
        self.passed_players = [False, False, False, False]
        # controlling_player stays the same (they won the round and start the next)
        
    def record_play(self, player_id: int, combo: Combo):
        """Record a played combo and update state. Assumes the play is already validated."""
        if combo.type == ComboType.PASS:
            self.passed_players[player_id] = True
            # Check for end of round
            if sum(self.passed_players) >= 3:
                self.end_round()
        else:
            self.current_combo = combo
            self.controlling_player = player_id
            self.is_first_turn = False
            
            # If a player plays out of turn (e.g., chopping with 4 pairs), they are back in the round
            if self.passed_players[player_id]:
                self.passed_players[player_id] = False
                
            # Update history and counts
            for card in combo.cards:
                self.played_cards[card] = 1
            self.player_card_counts[player_id] -= combo.length
            
    def get_observation_vector(self, observing_player_id: int, hand_cards: List[int]) -> List[float]:
        """
        Build the observation vector for RL agent. Extremely important for Card Counting.
        1. Hand Cards (52 binary)
        2. Played Cards History (52 binary)
        3. Opponent counts (3 floats normalized to [0, 1])
        4. Current Combo Type (one-hot 8 dims)
        5. Current Combo Highest Card (52 binary)
        6. Status Flags (2 binary)
        Total size: 52 + 52 + 3 + 8 + 52 + 2 = 169 dimensions.
        """
        obs = []
        
        # 1. Hand Cards
        hand = [0.0] * 52
        for c in hand_cards:
            hand[c] = 1.0
        obs.extend(hand)
        
        # 2. Played Cards Memory (Card Counting Core)
        obs.extend([float(x) for x in self.played_cards])
        
        # 3. Opponent counts 
        # Order: Right, Across, Left
        opp1 = (observing_player_id + 1) % 4
        opp2 = (observing_player_id + 2) % 4
        opp3 = (observing_player_id + 3) % 4
        
        obs.append(self.player_card_counts[opp1] / 13.0)
        obs.append(self.player_card_counts[opp2] / 13.0)
        obs.append(self.player_card_counts[opp3] / 13.0)
        
        # 4. Table Combo Type
        combo_type = [0.0] * 8
        if self.current_combo:
            combo_type[self.current_combo.type] = 1.0
        else:
            combo_type[ComboType.PASS] = 1.0 # Table is empty
        obs.extend(combo_type)
        
        # 5. Table Highest Card
        highest_card = [0.0] * 52
        if self.current_combo and self.current_combo.highest_card >= 0:
            highest_card[self.current_combo.highest_card] = 1.0
        obs.extend(highest_card)
        
        # 6. Status Flags
        is_free_turn = 1.0 if not self.current_combo else 0.0
        must_play_3_spades = 1.0 if self.is_first_turn else 0.0
        
        obs.append(is_free_turn)
        obs.append(must_play_3_spades)
        
        return obs
