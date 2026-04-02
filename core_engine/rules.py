from typing import List, Optional
from core_engine.deck import get_value, get_suit

class ComboType:
    INVALID = -1
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    STRAIGHT = 4
    THREE_PAIRS = 5
    QUAD = 6
    FOUR_PAIRS = 7

class Combo:
    def __init__(self, cards: List[int]):
        """Cards will be automatically sorted by ID"""
        self.cards = sorted(cards)
        self.type = ComboType.INVALID
        self.length = len(self.cards)
        self.highest_card = -1 if not self.cards else self.cards[-1]
        
        if self.length == 0:
            self.type = ComboType.PASS
        else:
            self._evaluate()
            
    def _evaluate(self):
        if self.length == 1:
            self.type = ComboType.SINGLE
            return
            
        values = [get_value(c) for c in self.cards]
        unique_values = set(values)
        
        if self.length == 2:
            if len(unique_values) == 1:
                self.type = ComboType.PAIR
            return
            
        if self.length == 3:
            if len(unique_values) == 1:
                self.type = ComboType.TRIPLE
            elif self._is_straight(values):
                self.type = ComboType.STRAIGHT
            return
            
        if self.length == 4:
            if len(unique_values) == 1:
                self.type = ComboType.QUAD
            elif self._is_straight(values):
                self.type = ComboType.STRAIGHT
            return

        if self.length >= 3 and self._is_straight(values):
            self.type = ComboType.STRAIGHT
            return
            
        if self.length == 6 and self._is_pair_sequence(values, 3):
            self.type = ComboType.THREE_PAIRS
            return
            
        if self.length == 8 and self._is_pair_sequence(values, 4):
            self.type = ComboType.FOUR_PAIRS
            return
            
    def _is_straight(self, values: List[int]) -> bool:
        # No 2 allowed in straight
        if 12 in values:
            return False
        # Must be consecutive. Cards are sorted by ID. 
        # Values must also be sorted because ID = value * 4 + suit.
        if len(set(values)) != len(values): # no duplicates allowed
            return False
        for i in range(len(values) - 1):
            if values[i+1] != values[i] + 1:
                return False
        return True
        
    def _is_pair_sequence(self, values: List[int], expected_pairs: int) -> bool:
        # No 2 allowed in pair sequences
        if 12 in values:
            return False
            
        # Values should appear exactly twice each, and be consecutive
        if len(set(values)) != expected_pairs:
            return False
            
        from collections import Counter
        counts = Counter(values)
        if any(c != 2 for c in counts.values()):
            return False
            
        unique_vals = sorted(list(set(values)))
        for i in range(len(unique_vals) - 1):
            if unique_vals[i+1] != unique_vals[i] + 1:
                return False
        return True

    def is_valid(self) -> bool:
        return self.type != ComboType.INVALID

def can_beat(current_combo: Combo, prev_combo: Optional[Combo]) -> bool:
    """Return True if current_combo can legitimately be played over prev_combo.
    This function handles both standard hierarchy and 'chặt heos/hàng' logic."""
    if not current_combo.is_valid():
        return False
        
    if current_combo.type == ComboType.PASS:
        return True

    if prev_combo is None or prev_combo.type == ComboType.PASS:
        return True
        
    # 1. Chop logic (Luật chặt Heo, Hàng)
    # prev is Single Heo (value == 12)
    if prev_combo.type == ComboType.SINGLE and get_value(prev_combo.highest_card) == 12:
        if current_combo.type in [ComboType.THREE_PAIRS, ComboType.QUAD, ComboType.FOUR_PAIRS]:
            return True
            
    # prev is Pair of Heos (2 cards value == 12)
    elif prev_combo.type == ComboType.PAIR and get_value(prev_combo.highest_card) == 12:
        if current_combo.type in [ComboType.QUAD, ComboType.FOUR_PAIRS]:
            return True
            
    # prev is Three Pairs (Ba đôi thông)
    elif prev_combo.type == ComboType.THREE_PAIRS:
        if current_combo.type in [ComboType.QUAD, ComboType.FOUR_PAIRS]:
            return True
            
    # prev is Quad (Tứ quý)
    elif prev_combo.type == ComboType.QUAD:
        if current_combo.type == ComboType.FOUR_PAIRS:
            return True

    # 2. Same type and length comparison
    if current_combo.type == prev_combo.type and current_combo.length == prev_combo.length:
        return current_combo.highest_card > prev_combo.highest_card
            
    return False

def can_play_out_of_turn(combo: Combo) -> bool:
    """True if this combo can be played even if the player has passed the current round"""
    # Only 4-pair sequence can drop out of turn to chop (Bốn đôi thông chặt không cần vòng)
    return combo.type == ComboType.FOUR_PAIRS
