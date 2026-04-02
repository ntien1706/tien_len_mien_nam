# BẢN THIẾT KẾ HỆ THỐNG: AI CHƠI TIẾN LÊN MIỀN NAM BẰNG REINFORCEMENT LEARNING (PPO & SELF-PLAY)

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG
Dự án được xây dựng dựa trên kiến trúc Modular, tách biệt hoàn toàn giữa **Game Engine** (Môi trường - Environment) chịu trách nhiệm mô phỏng luật chơi và **AI Agent** (Tác nhân học máy) chịu trách nhiệm ra quyết định. 

**Công nghệ & Framework sử dụng:**
- **Ngôn ngữ:** Python 3.10+.
- **RL Framework:** Stable-Baselines3-contrib (để tận dụng Maskable PPO) hoặc Ray RLlib.
- **Environment API:** Gymnasium (chuẩn OpenAI Gym).
- **CI/CD:** GitHub Actions (Automated Training & Model Checkpointing liên tục để không mất dữ liệu).

---

## 2. ENVIRONMENT & GAME LOGIC (TIẾN LÊN MIỀN NAM ENGINE)
Game Engine phải đảm bảo chặt chẽ 100% luật chơi Tiến Lên Miền Nam đã định nghĩa từ A-Z. Logic được tách riêng để kiểm tra dễ qua Unit Test.

### 2.1. Mã hóa bộ bài (Card Encoding)
Sử dụng ID từ 0 đến 51 để biểu diễn 52 lá bài.
- **Giá trị (Value):** `3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A, 2` tương ứng với các giá trị `0 -> 12`.
- **Chất (Suit):** `Bích (0), Chuồn (1), Rô (2), Cơ (3)`.
- **Công thức tính toán ID:** `Card_ID = Value * 4 + Suit`.
  - *Ví dụ:* 3♠ = 0 * 4 + 0 = 0. 2♥ = 12 * 4 + 3 = 51.
- **Lợi ích:** Mã hóa này giúp việc so sánh trực tiếp 2 lá bài cực kỳ dễ dàng: `Card1 > Card2` khi và chỉ khi `ID1 > ID2`.

### 2.2. Kiểm tra bộ bài hợp lệ (Combo Validators)
Hệ thống nhận vào danh sách các lá bài và phân loại thành các `Combo` sau:
- **Đơn (Single):** `len(cards) == 1`.
- **Đôi (Pair):** `len(cards) == 2` & cùng Value. Đôi lớn hơn tính theo Suit cao nhất (ID cao nhất).
- **Tam (Sám cô):** `len(cards) == 3` & cùng Value.
- **Sảnh (Straight):** `len(cards) >= 3`. Kiểm tra `Value` có liên tiếp hay không. Định nghĩa giới hạn:
  - **KHÔNG CHỨA LÁ 2:** Nếu trong danh sách kiểm tra có lá VALUE == 12 thì trả về Invalid. Sảnh lớn nhất mặc định là Q-K-A (Value 9-10-11). So sánh các sảnh đồng độ dài dựa trên `Card_ID` của lá cuối cùng.
- **Ba đôi thông:** 6 lá bài là tổ hợp của 3 đôi có Value liên tiếp. Bắt buộc không được chứa 2.
- **Bốn đôi thông:** 8 lá bài là tổ hợp của 4 đôi có Value liên tiếp. Bắt buộc không được chứa 2.
- **Tứ Quý:** 4 lá bài có cùng Value.

### 2.3. Logic Vòng Chơi & Trạng Thái Bàn (Table State)
- **Bắt đầu ván (Start Game):** Tìm ra người cầm 3♠ (ID = 0) đánh trước. Bắt buộc `Combo` tung ra đầu tiên phải chứa ID 0.
- **Quản lý Lượt (Turn Tracking):** Hệ thống có mảng `passed_players = [False, False, False, False]`. Ai bỏ lượt sẽ tính là True trong VÒNG đó. 
- **Chặn bài Đồng cấp:** Kiểm tra `Type` của bài đánh ra với `Type` trên bàn hiện tại (Đơn chặn Đơn, Đôi chặn Đôi...). Số lượng lá bài phải khớp nhau. Lá quyết định của người đi sau phải có ID cao hơn người trước.
- **Kết thúc vòng:** Khi 3 người đánh dấu là `passed = True`, người `False` cuối cùng sẽ giữ cái, gạt bài cũ trên bàn dọn đi (Bàn rỗng), reset `passed_players` và người này được đánh Type bài tự do.
- **Đặc tả riêng luật CHẶT HEO/HÀNG (Cực kỳ nghiêm ngặt):**
  - **Trên bàn có 1 Heo:** Chặt được bởi Heo lớn hơn, Ba đôi thông, Tứ Quý, Bốn đôi thông.
  - **Trên bàn có Đôi Heo:** Chặt được bởi Tứ Quý, Bốn đôi thông.
  - **Trên bàn có Ba đôi thông:** Chặt được bởi Ba đôi thông lớn hơn, Tứ quý, Bốn đôi thông.
  - **Trên bàn có Tứ Quý:** Chặt được bởi Tứ Quý lớn hơn, Bốn đôi thông.
  - **Trên bàn có Bốn đôi thông:** Bốn đôi thông lớn hơn.
  - **Luật không cần vòng:** Nếu `Combo(Bốn đôi thông)` được kích hoạt, cho phép bỏ qua trạng thái `passed = True` của người chơi. Tức là dù đã quăng bài bỏ vòng trước đó, bị đối phương dằn Heo/Hàng vẫn được nhảy vào chặt như thường.
- **Luật kết thúc (Endgame):** Game dừng khi có 1 người còn 0 lá bài trên tay.
  - Phạt "Thối": Duyệt Array bài dư trên tay 3 người thua, phát hiện Heo (Value 12) hoặc Hàng (Tứ quý/Đôi thông) áp dụng tiền phạt.
  - Phạt "Cóng": Người thua còn y nguyên 13 lá bị đè tiền phạt sâu nhất.

---

## 3. KHÔNG GIAN TRẠNG THÁI (STATE SPACE - CORE CARD COUNTING)
Để AI trở thành bậc thầy "hiểu bài" và "nhớ bài", State cung cấp cho NN phải là một Multi-Binary / Flat Vector cực kỳ đầy đủ thông tin để thuật toán tự diễn dịch ra Strategy.

Vector Trạng thái định vị `(Shape ~ 250 - 300 vector elements)`:
1. **Quản lý Bài Của Mình (Hand Cards):** Flat Binary Vector `[1 x 52]`. Vị trí ID tương ứng trên mảng sẽ bằng `1` (đang có bài cầm trên tay), ngõ ra `0` (không có).
2. **Bộ Nhớ Đã Đánh (Played Cards History):** Binary Vector `[1 x 52]`. Ghi chú tất cả những lá bài nào của tát cả user từng hạ xuống bàn (`1`: đã rời tay, `0`: chưa hiện hình). Vô cùng giá trị cho việc đếm bao nhiêu con Heo đỏ còn lại, bao nhiêu quân Át chưa xuất hiện.
3. **Tracking Bài Giấu Của Đối Thủ (Opponent Card Count):** Array `[3 x 1]`. Chứa số lượng bài còn lại trên tay của từng đối tác (từ 0 -> 13 normalized). AI sẽ dựa vào đây lôi bài rác ra để ép đối thủ chuẩn bị hết bài.
4. **Trạng thái Mâm Bài Hiện Tại (Table Context):**
   - Loại bài trên mâm đang đón đợi: Vector One-hot cho loại Combo (Trống, Đơn, Đôi, Sảnh 3, Sảnh 5, Ba Đôi Thông...).
   - Lá bài To Nhất Đè Mâm hiện tại: Binary Vector One-hot phân giải 52 nhịp. 
5. **Ownership Status:** Flag Binary `1/0`: Tôi có phải là chủ bàn không (Đạt cái chưa).

---

## 4. KHÔNG GIAN HÀNH ĐỘNG & ACTION MASKING (VẤN ĐỀ SINH TỬ)
Không gian tổ hợp (Action Space) của Tiến Lên vô cùng phức tạp, nếu cho sinh ngẫu nhiên mạng Nơ-ron sẽ sinh ra hàng tỷ chuỗi rác dẫn đến mô hình không bao giờ hội tụ.

### 4.1. Static Action Space (Action Manager)
Cần ánh xạ toàn bộ mọi tổ hợp thành một mã số Action. Phải viết riêng một module `action_manager.py` với class `ActionManager` để "dịch" (mapping/unmapping) qua lại giữa một Action ID (ví dụ số 5432) và một bộ bài cụ thể (ví dụ Sảnh 3-4-5). Đây là khâu then chốt xử lý sự phức tạp của không gian hành động.
Ta định nghĩa các tổ hợp theo chuẩn Combinations hợp lệ:
- Action `0`: Bỏ Lượt (Pass).
- Các Action Đơn: 52 idx.
- Các Action Đôi: 78 idx.
- Các Action Tam: 52 idx.
- Các Action sảnh (khử Heo) theo tất cả tổ hợp độ dài (từ 3 đến 12).
- Action Đôi thông, Action Tứ quý.
=> Tổng cộng Action Space sẽ cố định làm một dạng Multi-Discrete kéo dài khoảng ~`10,000` đến `15,000` actions khả dụng.

### 4.2. Action Masking Component
Action Masking là BẮT BUỘC để giới hạn khả năng NN trả ra Illegal Move. Trong thư viện SB3 Contrib (`MaskablePPO`):
- Chúng ta cung cấp hàm `get_action_mask()`.
- Hàm này trích lọc trạng thái `Trạng Thái Tay Bài` + `Định dạng Bàn Hiện Tại`. Nó xuất ra 1 mảng MASK `[10000] array(Boolean)`. 
- **Quy tắc Trích Bật Mask (True):** Mảng Boolean chỉ lấy `True` khi Type combo thỏa luật đánh đè hiện trường và Các lá bài trong Combo yêu cầu PHẢI CÓ trong tập tay bài đang sở hữu của User (Hand Cards). Các vị trí `False` bị gán `-Infinity Logits` trước khi lớp Softmax của AI đưa qua mạng. Từ đó Model tự do explore trong phạm vi "luôn luôn chơi đúng luật".

---

## 5. HÀM TRẢ THƯỞNG (REWARD FUNCTION)
Nghệ thuật cốt lõi giúp AI hành động tối ưu như một bài thủ thực thụ là định hình hàm `Reward`:

- **A. Terminal Reward (Game End - Ưu tiên hàng đầu):**
  - Trở thành vị trí số 1 (Hết bài đầu tiên): `+200.0`.
  - Ai là kẻ thua cuộc còn lại: `-50.0`. Thua mà bị "Cóng" (Chưa rụng lá nào): Bị phạt cực nặng `-150.0`. Khuyến khích xả rác cho dù thua.
- **B. Event-driven Hack/Cut (Thưởng Nóng Tức Thời trên Engine):**
  - Thực hiện một cú chặt Heo (Heo đen `+10`, Heo đỏ `+20`).
  - Gây ra một cú sập Hàng (Tứ quý đè Ba đôi thông, Bốn đôi đè Tứ Quý): `+30.0`
  - Nếu bản thân BỊ CHẶT / MƯỢN ĐƯỜNG sẽ áp dụng điểm Negative Reward (`-10 / -20`).
  - Thối Heo/Hàng khi Over: Trừ thẳng tiền (`-15.0`) vào tổng kết. Tiền phạt này có thể bơm một phần ngược về cho Người về Nhất để định hình AI theo cấu trúc Tối đa hoá Return Profit.
- **C. Micro-Speed Shaping (Định hình tốc độ xả bài):**
  - Cho đi qua 1 tổ hợp thành công trót lọt (trường hợp không đả động gì tới heo): Tính reward nhẹ `+1.0`. Khuyến khích Agent đánh càng nhiều bài vứt ra khỏi tay càng tốt.
  - Tặng thêm phần thưởng nếu kết thúc nhịp ván cực kỳ sớm (13 lá mình = 0, chênh lệch lá đối phương càng nhiều càng rinh mức Thưởng Margin lớn).

---

## 6. THUẬT TOÁN: PPO & SELF-PLAY SETUP TỐI ƯU
Thuật toán lựa chọn tiến quyết là **MaskablePPO** từ thư viện **Stable-Baselines3-Contrib**.
- **Mô hình Agent:** Cấu trúc League Training. 4 Agent ngồi 4 cạnh bàn dùng ĐỒNG THỜI một Share Policy Network.
- **Kiến trúc mạng (Network Architecture):** Do giới hạn của GitHub Actions (không có GPU miễn phí), việc training với ~15,000 classes output sẽ nặng nề. Vì vậy, BẮT BUỘC sử dụng kiến trúc **MLP (Multi-Layer Perceptron)** đơn giản (chỉ vài lớp Fully Connected, ví dụ: `[256, 256, 256]`), tuyệt đối không dùng các cấu trúc cồng kềnh như Transformer hay CNN để đảm bảo tốc độ epoch chạy đủ nhanh trên CPU.
- **Tại sao gọi là Self-Play:** Nó tự chia bài với chính mình (Clones). Sau mỗi vòng, weights liên tục update, dẫn đến việc phải đối đầu với đối tượng (chính mình) bản vá mới nhất. 
- **Kỹ thuật chống Quên Local Minima (Forgetting Mitigation):** Trong các trận Self-play, ngoài việc Load trọng số hiện tại đánh 4 bên, thỉnh thoảng `Agent Đối Thủ` sẽ lấy Random History Checkpoint Weights (Ví dụ: Model đã train tại 2 ngày trước) để Model hiện trường giữ vững tính linh hoạt thích ứng thay vì chỉ hiểu mỗi chiến thuật của một bản.

---

## 7. CLI ASSISTANT MODE (ỨNG DỤNG BÁO BÀI - LÀM TRỢ LÝ)
Một tính năng ứng dụng từ mạng Model thu về, chuyên biệt dành cho User.
- **Cơ chế Input Data:** Người dùng điền array bài thật của mình thông qua Text CLI (`User> Nhập tay bài: 3H 4S 5H 7C 7S 8S 10D JC JC QH KS AD 2D`). Game engine ảo tạo State ngay lập tức.
- **Cơ chế Trinh Sát Loop:**
  - Đối tác đi bài `P_right> 3C 4C 5C`. Script cập nhật Trạng Thái Bàn.
  - User dùng lệnh `:suggest` để kêu gọi ý kiến. Agent nạp dữ liệu này, đưa qua mạng tính toán với `get_action_mask`. Chọn ra TOP Logits cao nhất. Màn hinh in:
    => *"🤖 AI Gợi ý: HÃY BỎ LƯỢT NHÉ! (Tỉ lệ Win: 84%). Đối thủ cầm số lượng bài lớn nhưng bộ này không làm hại tới cấu trúc. Gợi ý 2: Chặt (7S 8S... - 12%)"*
- Chức năng chứng minh tường minh khả năng Card Counting và đưa hệ thống lên đẳng cấp của một Assistant thay vì chỉ chơi mù tự động.

---

## 8. TÍCH HỢP CI/CD (GITHUB ACTIONS INTEGRATION)
Hệ thống CI/CD đảm đương việc Treo máy Train trên Cloud cực kỳ quan trọng do RL tốn thời gian tính toán diện rộng. Giải quyết vấn đề VM Cloud đóng gây mất checkpoint.

- **Workflow Script (`.github/workflows/training_loop.yml`):**
  1. Trigger bằng `schedule` (Mỗi đêm 0h00) hoặc `workflow_dispatch`.
  2. Setup Node VM / Ubuntu Runner. Install các Depenedencies (Pytorch, sb3).
  3. Gọi Python Script `train.py`. Script này cho chạy Limit `5.000.000` Timesteps hoặc Thời lượng hẹn giờ thoát (Ví dụ Timer chạm 5 tiếng rưỡi thì Safe Stop). Cất Model đuôi `.zip` qua folder Output.
  4. Automation Pipeline: Gọi Action plugin `stefanzweifel/git-auto-commit-action`. Git script tự động Add Tracking File, ghi Commit tự động `Automated Model Checkpoint - [Date]`, và `git push` đè nhánh main lưu về. Vòng lặp ngắt điện an toàn 100%.

---

## 9. CẤU TRÚC CODE THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```txt
tien_len_mien_nam_ai/
│
├── .github/
│   └── workflows/
│       └── autosave_training.yml   # CI/CD Tự kích hoạt train và auto commit
│
├── core_engine/
│   ├── __init__.py
│   ├── rules.py                    # Khung sườn Logic tĩnh (Xác thực Combo, Luật chặn)
│   ├── deck.py                     # Quản lý 52 lá, Shuffle, Phân rã Card ID
│   └── tracker.py                  # Dựng History, Card Counting, Đếm bài thừa
│
├── rlenv/
│   ├── __init__.py
│   ├── action_manager.py           # Phân dịch các mã Action ID (từ 0 đến ~15000) ngược về bộ bài cụ thể
│   ├── tienlen_gym_v1.py           # Kế thừa gym.Env. Map Action Space và Vectorize Observation.
│   └── action_masker.py            # Layer xử lý che lấp Invalid Acts riêng biệt
│
├── trained_models/
│   ├── best_agent.zip              # Lưu Model tốt nhất qua Evaluation
│   └── checkpoints/                # Auto-save qua từng epoch
│
├── scripts/
│   ├── train.py                    # Điểm khởi tạo Multi-Agent Self-Play 
│   ├── evaluate.py                 # Bot match (Random vs Model, Model cũ vs Mới) tỷ lệ Elo
│   └── cli_assistant.py            # Prompt giao diện trợ lý nhắc bài User tại thời gian thực
│
├── requirements.txt
├── README.md
└── PLAN.md                         # File bản thiết kế kiến trúc AI này
```
