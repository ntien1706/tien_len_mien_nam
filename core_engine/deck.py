import random
from typing import List, Tuple

# Card Encoding:
# Values: 3=0, 4=1, ..., K=10, A=11, 2=12
# Suits: Bích (Spades, ♠)=0, Chuồn (Clubs, ♣)=1, Rô (Diamonds, ♦)=2, Cơ (Hearts, ♥)=3
# Card ID = Value * 4 + Suit

SUITS = ['bi', 'ch', 'ro', 'co']
VALUES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']

def get_value(card_id: int) -> int:
    return card_id // 4

def get_suit(card_id: int) -> int:
    return card_id % 4

def card_to_string(card_id: int) -> str:
    if card_id < 0 or card_id > 51:
        return "Unknown"
    v = get_value(card_id)
    s = get_suit(card_id)
    return f"{VALUES[v]}{SUITS[s]}"

def cards_to_string(cards: List[int]) -> str:
    if not cards:
        return "Pass"
    return " ".join([card_to_string(c) for c in sorted(cards)])

def string_to_card(card_str: str) -> int:
    """
    Parse a string like '3♠', '3S' (Spades), '2H' (Hearts), '3bi' into card ID
    """
    card_str = card_str.upper().strip()
    if not card_str:
        return -1
        
    suit_char = card_str[-1]
    value_str = card_str[:-1]
    
    suit_map = {
        'S': 0, '♠': 0, 'B': 0,'BI': 0, # Bích
        'C': 1, '♣': 1, 'CH': 1, # Chuồn
        'D': 2, '♦': 2, 'R': 2, 'RO': 2, # Rô
        'H': 3, '♥': 3, 'CO': 3 # Cơ
    }
    
    # Handle multi-char suits
    for k, v in suit_map.items():
        if card_str.endswith(k) and len(k) > 1:
            suit_char = k
            value_str = card_str[:-len(k)]
            break
            
    if suit_char not in suit_map:
        return -1
        
    suit = suit_map[suit_char]
    
    if value_str == 'AT': value_str = 'A' # Át
    
    if value_str not in VALUES:
        return -1
        
    value = VALUES.index(value_str)
    return value * 4 + suit

def string_to_cards(cards_str: str) -> List[int]:
    if cards_str.lower().strip() == "pass":
        return []
    parts = cards_str.split()
    return [string_to_card(p) for p in parts if string_to_card(p) != -1]

class Deck:
    def __init__(self):
        self.cards = list(range(52))
        
    def shuffle(self):
        random.shuffle(self.cards)
        
    def deal(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """Deals 13 cards to 4 players. Sorts hands."""
        self.shuffle()
        p1 = sorted(self.cards[0:13])
        p2 = sorted(self.cards[13:26])
        p3 = sorted(self.cards[26:39])
        p4 = sorted(self.cards[39:52])
        return p1, p2, p3, p4
