import os
import sys

# Đảm bảo import được các module từ thư mục dự án
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from rlenv.tienlen_gym_v1 import TienLenEnv

MODEL_PATH = os.path.join(project_root, "trained_models", "best_agent")
CHECKPOINT_DIR = os.path.join(project_root, "trained_models", "checkpoints")

def ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_self_play(total_timesteps: int = 1000000, time_limit_hours: float = 5.5):
    ensure_dirs()
    
    # Kích hoạt Môi trường Tiền Lên
    env = TienLenEnv(opponent_policy=None)
    
    if os.path.exists(MODEL_PATH + ".zip"):
        print("[*] Đang tải Model cũ để tiếp tục huấn luyện tăng cường...")
        model = MaskablePPO.load(MODEL_PATH, env=env)
    else:
        print("[+] Tạo Model MaskablePPO mới (Kiến trúc MLP 256x256x256 để tối ưu CPU)...")
        # Sử dụng Mạng MLP đơn giản phù hợp với Action Space ~408 và chạy cực mượt trên CPU
        policy_kwargs = dict(net_arch=[256, 256, 256])
        
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            gamma=0.99
        )
        
    chunk_timesteps = min(50000, total_timesteps)
    epochs = total_timesteps // chunk_timesteps
    
    print(f"=== Bắt đầu Vòng lặp League Self-Play ({epochs} epochs) ===")
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Train Agent 0
        model.learn(total_timesteps=chunk_timesteps, reset_num_timesteps=False)
        
        # Save Checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"agent_epoch_{epoch+1}")
        model.save(checkpoint_path)
        
        # !Self-play trick:
        # Cập nhật não bộ Đối thủ thành Model mới nhất mà ta vừa train được!
        # Như vậy AI sẽ luôn phải đấu với một phiên bản mạnh hơn sau mỗi chunck.
        print("=> Cập nhật Trọng số Model cho Đối Thủ...")
        env.opponent_policy = model
        
        model.save(MODEL_PATH)
        
        # Check time limit for Graceful Shutdown
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit_hours * 3600:
            print(f"\n[!] Đã đạt giới hạn thời gian chạy {time_limit_hours} giờ. Kích hoạt quá trình Lưu & Dừng an toàn (Graceful Shutdown)!")
            break
            
    duration = time.time() - start_time
    print(f"\n[!] Training đã xong sau {duration:.2f} giây.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5000000, help="Tổng sô timesteps cho quá trình học")
    parser.add_argument("--time-limit", type=float, default=5.5, help="Giới hạn thời gian chạy (giờ) trước khi quá trình tự động dừng (graceful shutdown)")
    args = parser.parse_args()
    
    train_self_play(args.timesteps, args.time_limit)
