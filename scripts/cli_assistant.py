import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from sb3_contrib import MaskablePPO

from core_engine.deck import cards_to_string, string_to_cards
from core_engine.tracker import GameTracker
from core_engine.rules import Combo
from rlenv.action_manager import ActionManager
from rlenv.action_masker import ActionMasker

MODEL_PATH = os.path.join(project_root, "trained_models", "best_agent.zip")

def main():
    print("========================================")
    print("   TIẾN LÊN MIỀN NAM - AI ASSISTANT     ")
    print("========================================")
    
    if not os.path.exists(MODEL_PATH):
        print("[!] Không tìm thấy file model đã train tại", MODEL_PATH)
        print("[!] Bạn cần chạy lệnh `python scripts/train.py` (hoặc push lên GitHub) để có model.")
        return
        
    print("[*] Đang tải Bộ não Trí Tuệ Nhân Tạo...")
    model = MaskablePPO.load(MODEL_PATH)
    
    action_manager = ActionManager()
    masker = ActionMasker(action_manager)
    tracker = GameTracker()
    
    print("\nQuy ước nhập - Cực kỳ dể hiểu:")
    print("- Bài trên tay: Nhập các lá cách nhau khoảng trắng. Ví dụ: '3bi 4chuon 5ro Aco 2bi'")
    print("- Ký hiệu chất: Rô(r, ro), Cơ(co), Chuồn(c, ch), Bích(b, bi). Át(A).")
    print("- Bỏ lượt: Gõ chữ 'pass'")
    
    hand_input = input("\n[USER] Khai báo toàn bộ bài bạn đang cầm: ")
    hand_cards = string_to_cards(hand_input)
    print(f"-> Ghi nhận {len(hand_cards)} lá: {cards_to_string(hand_cards)}")
    
    user_id = 0 # Ta luôn đứng vị trí Player 0 để quan sát
    tracker.passed_players = [False, False, False, False]
    
    while True:
        print("\n" + "="*30)
        print("--- THÔNG TIN CHIẾN SỰ MÂM BÀI ---")
        if tracker.current_combo:
            print(f"-> Mâm đang bị đè bởi: {cards_to_string(tracker.current_combo.cards)}")
            print(f"-> Tác giả: Player {tracker.controlling_player} (Bạn là P0)")
        else:
            print("-> Mâm đang trống (Quyền đi tự do)")
            
        print("\nHÀNH ĐỘNG CỦA BẠN:")
        print("1. [TRINH SÁT] Nhập bài một Đối thủ vừa đánh")
        print("2. [NHỜ TRỢ GIÚP] Kênh kết nối với Model RL")
        print("3. [XUẤT QUÂN] Báo đã kết thúc lượt đi của mình")
        print("4. [XONG VÒNG] Một người vừa báo 'Bỏ lượt' hoặc mâm rỗng")
        print("0. Thoát")
        
        choice = input("Mời chọn (0-4): ")
        
        if choice == '1':
            cards_str = input("Nhập bài đối thủ đánh ra: ")
            cards = string_to_cards(cards_str)
            if not cards: continue
            pid = input("Ai đánh? (1 = Người sau bạn, 2 = Trước mặt, 3 = Trước bạn): ")
            try: pid = int(pid) % 4
            except: pid = 1
            tracker.record_play(pid, Combo(cards))
            print(f"[+] Hệ thống Card Counting ghi nhận: P{pid} xả {len(cards)} lá.")
            
        elif choice == '2':
            obs = tracker.get_observation_vector(user_id, hand_cards)
            mask = masker.get_action_mask(hand_cards, tracker, user_id)
            
            valid_action_indices = np.where(mask)[0]
            if len(valid_action_indices) == 0:
                print("\n🤖 AI NÓI RẰNG: THUA RỒI BẠN ƠI, CHIẾN THUẬT LÀ BỎ LƯỢT!")
                continue
                
            dist = model.policy.get_distribution(np.array([obs])).distribution
            logits = dist.logits.detach().numpy()[0]
            
            # Khôi phục mảng Logits thành Tỷ lệ % của các nước đi đúng luật
            valid_logits = logits[valid_action_indices]
            exp_logits = np.exp(valid_logits - np.max(valid_logits)) # softmax safe
            valid_probs = exp_logits / np.sum(exp_logits)
            
            sorted_idx = np.argsort(valid_probs)[::-1]
            
            print("\n🤖 === BỘ NÃO RL PHÂN TÍCH === 🤖")
            for i, idx in enumerate(sorted_idx[:3]): # Top 3 gợi ý
                real_action_id = valid_action_indices[idx]
                prob = valid_probs[idx] * 100
                
                cards_to_play = action_manager.decode_action(real_action_id, hand_cards)
                cards_str = cards_to_string(cards_to_play) if cards_to_play else "BỎ LƯỢT CẤT CHỨA BÀI (Pass)"
                
                if i == 0:
                    print(f"⭐ BEST MOVE (Khuyến nghị số 1): Hãy ném ra {cards_str}")
                    print(f"   => Win-rate Confidence: {prob:.1f}%")
                else:
                    print(f"   - Lựa chọn thay thế {i+1}: {cards_str} ({prob:.1f}%)")
                    
        elif choice == '3':
            cards_str = input("Bạn đã chốt vất lá gì ra bàn? (Ghi bài hoặc gõ 'pass'): ")
            cards = string_to_cards(cards_str)
            tracker.record_play(user_id, Combo(cards))
            # Hủy bài khỏi tay
            for c in cards:
                if c in hand_cards: hand_cards.remove(c)
            print(f"[+] Bài dư trên tay: {cards_to_string(hand_cards)}")
            
        elif choice == '4':
            pid = input("Ai vừa xin qua vòng? (1, 2, 3 - hoặc 0 nếu là bạn): ")
            try: pid = int(pid) % 4
            except: pid = 1
            tracker.record_play(pid, Combo([]))
            print(f"[+] Trạng thái đã ghi nhận.")

        elif choice == '0':
            break

if __name__ == "__main__":
    main()
