from typing import List, Tuple, Dict
from core_engine.deck import get_value, get_suit
from core_engine.rules import ComboType, Combo

class ActionManager:
    """
    Giảm thiểu không gian hành động (Action Space) từ hàng triệu combination 
    xuống còn một không gian "Intent-based" (~408 actions) để PPO chạy vượt trội trên CPU.
    
    Định nghĩa một action dựa trên 3 yếu tố:
    (Loại Combo, Độ dài, ID của lá bài cao nhất quyết định combo)
    
    Với các lá bài không phải lá cao nhất (Ví dụ các lá rác kéo theo trong sảnh), 
    hệ thống sẽ tự động pick những lá có CHẤT NHỎ NHẤT đang có trên tay để tối ưu bài cho user.
    """
    
    def __init__(self):
        self.actions: List[Tuple[int, int, int]] = []
        self._build_action_space()
        self.num_actions = len(self.actions)
        
    def _build_action_space(self):
        # 0. Pass
        self.actions.append((ComboType.PASS, 0, -1))
        
        for c_id in range(52):
            value = get_value(c_id)
            suit = get_suit(c_id)
            
            # 1. Single
            self.actions.append((ComboType.SINGLE, 1, c_id))
            
            # 2. Pair: requires at least 1 lower suit to form a pair with this highest card
            if suit >= 1:
                self.actions.append((ComboType.PAIR, 2, c_id))
                
            # 3. Triple: requires at least 2 lower suits
            if suit >= 2:
                self.actions.append((ComboType.TRIPLE, 3, c_id))
                
            # 4. Quad: requires all 3 lower suits
            if suit == 3:
                self.actions.append((ComboType.QUAD, 4, c_id))
                
            # 5. Straight (Sảnh): length 3 to 12. No Heo (value 12) allowed.
            if value >= 2 and value < 12: # Highest card value must be at least 5 (value index 2) to form a len 3 straight.
                max_len = value + 1 # if value is 2 (card 5), max len is 3 (3-4-5)
                for length in range(3, min(max_len + 1, 13)):
                    self.actions.append((ComboType.STRAIGHT, length, c_id))
                    
            # 6. Three Pairs (3 đôi thông): end value must be at least 5 (index 2). No Heo.
            if value >= 2 and value < 12 and suit >= 1:
                self.actions.append((ComboType.THREE_PAIRS, 6, c_id))
                
            # 7. Four Pairs (4 đôi thông): end value must be at least 6 (index 3). No Heo.
            if value >= 3 and value < 12 and suit >= 1:
                self.actions.append((ComboType.FOUR_PAIRS, 8, c_id))

    def decode_action(self, action_id: int, hand_cards: List[int]) -> List[int]:
        """Dịch Action ID thành một danh sách các lá bài cụ thể từ tay người chơi"""
        if action_id == 0:
            return []
            
        c_type, length, highest_card = self.actions[action_id]
        
        # Helper: tìm list bài trong tay có giá trị `val`
        def get_cards_of_value(val: int) -> List[int]:
            return [c for c in hand_cards if get_value(c) == val]
            
        if c_type == ComboType.SINGLE:
            return [highest_card] if highest_card in hand_cards else []
            
        if c_type == ComboType.PAIR:
            val = get_value(highest_card)
            available = get_cards_of_value(val)
            available = [c for c in available if c < highest_card]
            if not available or highest_card not in hand_cards: return []
            return [available[0], highest_card]
            
        if c_type == ComboType.TRIPLE:
            val = get_value(highest_card)
            available = get_cards_of_value(val)
            available = [c for c in available if c < highest_card]
            if len(available) < 2 or highest_card not in hand_cards: return []
            return [available[0], available[1], highest_card]
            
        if c_type == ComboType.QUAD:
            val = get_value(highest_card)
            available = get_cards_of_value(val)
            if len(available) < 4: return []
            return available # All 4 suits
            
        if c_type == ComboType.STRAIGHT:
            if highest_card not in hand_cards: return []
            result = [highest_card]
            high_val = get_value(highest_card)
            for i in range(1, length):
                val_needed = high_val - i
                available = get_cards_of_value(val_needed)
                if not available: return [] # Missing a link in straight
                result.append(available[0]) # Pick the lowest suit by default
            return sorted(result)
            
        if c_type == ComboType.THREE_PAIRS or c_type == ComboType.FOUR_PAIRS:
            if highest_card not in hand_cards: return []
            num_pairs = 3 if c_type == ComboType.THREE_PAIRS else 4
            result = []
            high_val = get_value(highest_card)
            
            for i in range(num_pairs):
                val_needed = high_val - i
                available_for_pair = get_cards_of_value(val_needed)
                if i == 0: # The highest pair must contain the highest_card
                    available_for_pair = [c for c in available_for_pair if c < highest_card]
                    if not available_for_pair: return []
                    result.extend([available_for_pair[0], highest_card])
                else:
                    if len(available_for_pair) < 2: return []
                    result.extend([available_for_pair[0], available_for_pair[1]])
            return sorted(result)
            
        return []
