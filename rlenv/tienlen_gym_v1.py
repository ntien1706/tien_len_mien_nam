import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core_engine.deck import Deck, get_value
from core_engine.tracker import GameTracker
from core_engine.rules import Combo, ComboType
from rlenv.action_manager import ActionManager
from rlenv.action_masker import ActionMasker

class TienLenEnv(gym.Env):
    """
    Gymnasium Environment cho Tiến Lên Miền Nam.
    Được thiết kế cho MaskablePPO. Môi trường luôn được nhìn từ góc độ của Player 0.
    Trong kỹ thuật Self-Play, opponent_policy sẽ được gán để Env tự mô phỏng lượt của 3 đối thủ.
    """
    def __init__(self, opponent_policy=None):
        super(TienLenEnv, self).__init__()
        self.action_manager = ActionManager()
        self.masker = ActionMasker(self.action_manager)
        
        # Action space: ~408 distinct discrete actions
        self.action_space = spaces.Discrete(self.action_manager.num_actions)
        
        # Observation space shape: 169 (Flat Vector)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(169,), dtype=np.float32)
        
        self.opponent_policy = opponent_policy
        self.reset_game_state()
        
    def reset_game_state(self):
        self.deck = Deck()
        self.hands = self.deck.deal()
        self.tracker = GameTracker()
        self.current_player = -1
        
        # Luật: Ai cầm 3 Bích (0) đi trước
        for i, hand in enumerate(self.hands):
            if 0 in hand:
                self.current_player = i
                break
                
        self.done = False
        self.accumulated_reward = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_game_state()
        
        # Tự động chơi cho Đối thủ cho đến khi tới lượt của Agent 0
        self._simulate_until_p0()
        
        return self._get_obs(), {}
        
    def action_masks(self) -> np.ndarray:
        return self.masker.get_action_mask(self.hands[0], self.tracker, 0)
        
    def _simulate_until_p0(self):
        # Trôi vòng lặp cho đến khi (lượt của Player 0) HOẶC (ván đấu kết thúc)
        while self.current_player != 0 and not self.done:
            obs = self.tracker.get_observation_vector(self.current_player, self.hands[self.current_player])
            mask = self.masker.get_action_mask(self.hands[self.current_player], self.tracker, self.current_player)
            
            if self.opponent_policy is not None:
                # Lấy Output từ Agent Đối thủ
                action, _ = self.opponent_policy.predict(np.array(obs), action_masks=mask, deterministic=False)
                action = int(action)
            else:
                # Random bot dựa trên Rule Based Mask
                valid_actions = np.where(mask)[0]
                if len(valid_actions) == 0:
                    action = 0 # Lỗi mạng? Phải Pass
                else: 
                    # Ưu tiên không bỏ lượt (Đánh ngẫu nhiên 1 lá)
                    non_pass = [a for a in valid_actions if a != 0]
                    if len(non_pass) > 0:
                        action = np.random.choice(non_pass)
                    else:
                        action = 0
                    
            self._apply_action(self.current_player, action)
            
            if not self.done:
                self._next_turn()

    def _apply_action(self, player_id: int, action_id: int):
        cards = self.action_manager.decode_action(action_id, self.hands[player_id])
        combo = Combo(cards)
        
        # Tính thưởng / phạt trước khi State bàn cập nhật
        if combo.type in [ComboType.THREE_PAIRS, ComboType.QUAD, ComboType.FOUR_PAIRS]:
            if player_id == 0:
                self.accumulated_reward += 20.0 # Ta chặt Heo / chặt Hàng của máy!
            elif self.tracker.controlling_player == 0:
                self.accumulated_reward -= 20.0 # Oops, Agent 0 bị đối thủ đè Heo!
                
        self.tracker.record_play(player_id, combo)
        
        if combo.type != ComboType.PASS:
            for c in cards:
                self.hands[player_id].remove(c)
                
            if player_id == 0:
                self.accumulated_reward += 0.5 # Shaping Reward: Xả được bài
                
        # Kiểm tra điều kiện End Game
        if len(self.hands[player_id]) == 0:
            self.done = True
            if player_id == 0:
                # Ta về nhất, cộng Thưởng Margin (Chênh lệch)
                leftover_cards = sum([len(h) for h in self.hands[1:4]])
                self.accumulated_reward += 200.0 + (leftover_cards * 5.0) 
            else:
                # Đối thủ về nhất
                self.accumulated_reward -= 50.0
                if len(self.hands[0]) == 13:
                    self.accumulated_reward -= 150.0 # Bị cóng!
                heos = sum([1 for c in self.hands[0] if get_value(c) == 12])
                self.accumulated_reward -= (heos * 15.0) # Thối Heo
                
    def _next_turn(self):
        for _ in range(4):
            self.current_player = (self.current_player + 1) % 4
            if not self.tracker.passed_players[self.current_player]:
                break
                
    def step(self, action: int):
        self.accumulated_reward = 0.0
        
        mask = self.action_masks()
        if not mask[action]:
            # Đề phòng Agent xuất ra invalid do Bug
            return self._get_obs(), -500.0, True, False, {"error": "invalid_move_punishment"}
            
        self._apply_action(0, action)
        
        if not self.done:
            self._next_turn()
            self._simulate_until_p0()
            
        return self._get_obs(), self.accumulated_reward, self.done, False, {}
        
    def _get_obs(self):
        obs = self.tracker.get_observation_vector(0, self.hands[0])
        return np.array(obs, dtype=np.float32)
