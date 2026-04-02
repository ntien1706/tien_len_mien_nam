import numpy as np
from typing import List
from rlenv.action_manager import ActionManager
from core_engine.tracker import GameTracker
from core_engine.rules import Combo, can_beat, can_play_out_of_turn

class ActionMasker:
    """
    Tạo Mask để chặn AI chọn những action rác (không có bài trên tay, hoặc sai luật chặn bàn)
    Mask = True là hợp lệ, Mask = False sẽ bị Softmax đưa về âm vô cực.
    """
    def __init__(self, action_manager: ActionManager):
        self.action_manager = action_manager

    def get_action_mask(self, hand_cards: List[int], tracker: GameTracker, player_id: int) -> np.ndarray:
        mask = np.zeros(self.action_manager.num_actions, dtype=bool)
        
        is_passed = tracker.passed_players[player_id]
        table_combo = tracker.current_combo
        
        # Luật ván đầu tiên phải đánh 3 Bích (ID = 0)
        must_play_3_spades = tracker.is_first_turn
        
        for action_id in range(self.action_manager.num_actions):
            # 1. Thử ánh xạ Action ID -> Bài cầm trên tay
            cards = self.action_manager.decode_action(action_id, hand_cards)
            
            # Action 0 (Bỏ lượt)
            if action_id == 0:
                if not table_combo:
                    # Bàn trống (mình mướt vòng), bắt buộc phải phát bài, không được bỏ lượt
                    mask[0] = False
                elif must_play_3_spades:
                    # Bắt buộc phải đánh 3 bích, không được bỏ
                    mask[0] = False
                else:
                    mask[0] = True
                continue
                
            # Nếu decode failed (tức là không đủ bài trên tay để form combo đó)
            if not cards:
                continue
                
            # Kiểm tra luật 3 Bích
            if must_play_3_spades and 0 not in cards:
                continue
                
            # Tạo bộ combo và đánh giá tính hợp lệ
            combo = Combo(cards)
            if not combo.is_valid():
                continue
                
            # Nếu user đã Bỏ lượt ở các turn trước trong cùng VÒNG này
            if is_passed:
                # Chỉ được nhảy vô nếu có 4 đôi thông và đủ sức đè Heo/Hàng trên bàn
                if not can_play_out_of_turn(combo) or not can_beat(combo, table_combo):
                    continue
                else:
                    mask[action_id] = True
                    continue
                    
            # Lượt bình thường: So sánh combo mới với bài đang trên mâm
            if can_beat(combo, table_combo):
                mask[action_id] = True
                
        return mask
