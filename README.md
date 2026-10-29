# LLM-Lean: High-Fidelity Formal Theorem Proving Baseline

Cấu hình baseline tối ưu tuyệt đối cho **độ chính xác và độ ổn định logic** trên **1x GPU RTX 6000 Pro (96GB VRAM)**, kết hợp **Base weights BF16 nguyên bản** (không quantize) với **High-Rank RSLoRA ($r=128, \alpha=256$)** và quy trình huấn luyện 2 giai đoạn: **Cold-Start SFT** và **RLVR với GRPO**.

---

## 1. Thông số Kỹ thuật & Bảng Phân Bổ VRAM (96GB Budget)

| Thành phần | Cấu hình | Lý do kỹ thuật & Phân bổ VRAM |
| :--- | :--- | :--- |
| **Base Model Precision** | `bfloat16` (Unquantized) | Giữ nguyên 100% phân bố xác suất token và các cú pháp nhạy cảm của Lean 4 / Mathlib. Chiếm **~54 GB VRAM** (đối với model ~27B - 32B). |
| **Adapter Architecture** | **RSLoRA** ($r=128, \alpha=256$) | Scaling factor $1/\sqrt{r}$ của RSLoRA giúp model học sâu tương đương 98% full-tuning mà không bị bất ổn gradient. |
| **Target Modules** | All-Linear (`q, k, v, o, gate, up, down`) | Phủ toàn bộ attention và MLP layers để model nắm trọn cả semantic logic lẫn rule-based syntax. |
| **Optimizer** | `paged_adamw_8bit` | Chỉ tốn **~1.5 GB VRAM** và có tính năng tự động page sang host RAM nếu tràn đỉnh peak activation. |
| **Attention Backend** | `FlashAttention-2` | Tối ưu bộ nhớ cache và thông lượng tính toán chuỗi dài. |
| **Context Length ($L_{\max}$)** | 8,192 – 12,288 tokens | Đủ bao quát toàn bộ tactic trace, CoT dài (`<think>...</think>`) và log phản hồi từ Lean compiler. |
| **Gradient Checkpointing** | `True` (`use_reentrant: False`) | Giữ activation memory dưới **~16 GB** ở độ dài chuỗi 8k. |

### Tính toán VRAM thực tế (Stage 1 SFT):
$$\text{VRAM}_{\text{total}} = 54\text{ GB (Base)} + 2\text{ GB (RSLoRA + Opt)} + 16\text{ GB (Activation)} \approx \mathbf{72\text{ GB / } 96\text{ GB}}$$
*(Dư 24 GB an toàn tuyệt đối, loại bỏ 100% nguy cơ CUDA OOM).*

---

## 2. Lộ trình Huấn luyện Toàn diện 3 Giai đoạn (Full Pipeline)

```text
[Pre-trained Qwen3.8-27B] (BF16 Unquantized)
         │
         ▼
[Stage 1: Domain-Adaptive Continued Pre-training (DACP)]
   • Tri thức Lean 4 / Mathlib4, Textbooks Toán - Lý lý thuyết
   • Proof-Pile-2, OpenWebMath, FineMath, ArXiv Physics (quant-ph, math-ph)
         │
         ▼
[Stage 2: High-Quality Agentic & Formal SFT]
   • Synthetic Proof Search CoT + Tool Calling (REPL, Python, SymPy)
   • NuminaMath-TIR, NuminaMath-CoT, MathInstruct, Lean-Workbook
         │
         ▼
[Stage 3: RL with Verifiable Reward (RLVR / GRPO)]
   • Multi-Environment: Lean 4 REPL + Sandbox Python + Symbolic Math Solvers
   • Benchmarks: miniF2F-Lean4, ProofNet, Competition-MATH
```

```text
llm-lean/
├── configs/
│   ├── sft_config.yaml         # Cấu hình Cold-Start SFT
│   └── grpo_config.yaml        # Cấu hình RLVR GRPO + Verifier
├── data/
│   ├── lean4_sft_sample.jsonl  # Dữ liệu mẫu SFT (theorems, CoT, formal proofs)
│   └── lean4_rl_prompts.jsonl  # Prompts cho RL rollout
├── docker/
│   ├── Dockerfile.lean4        # Container Lean 4 + Mathlib cho CPU Sandbox
│   ├── docker-compose.yaml     # Cấu hình giới hạn 16-32 vCPU, 32GB RAM
│   ├── lakefile.lean           # Lean 4 project config with Mathlib
│   └── lean-toolchain          # Lean 4 stable toolchain version
├── scripts/
│   ├── run_sft.sh              # Bash script khởi chạy SFT
│   ├── run_grpo.sh             # Bash script khởi chạy GRPO
│   ├── setup_lean4.sh          # Script cài đặt Lean 4 & Mathlib trên host
│   └── test_verifier.py        # Kiểm thử độc lập Lean 4 Sandbox & Rewards
├── src/
│   ├── data/
│   │   ├── dataset.py          # Data loaders & chat templates
│   │   └── prompts.py          # System prompt & prompt formatters
│   ├── train/
│   │   ├── sft_train.py        # SFTTrainer execution
│   │   └── grpo_train.py       # GRPOTrainer + vLLM rollout
│   ├── utils/
│   │   └── gpu_mem.py          # Monitor VRAM & bộ nhớ GPU
│   └── verifier/
│       ├── lean4_repl.py       # Lean 4 compiler & REPL subprocess sandbox
│       ├── rewards.py          # Hàm reward chuẩn xác cho GRPO
│       └── sandbox_manager.py  # Quản lý CPU worker pool song song
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Quy trình Triển khai Chi tiết

### Bước 1: Cài đặt Thư viện Python

```bash
pip install -r requirements.txt
# Hoặc cài đặt flash-attn nếu tương thích CUDA:
pip install flash-attn --no-build-isolation
```

### Bước 2: Thiết lập CPU Sandbox Verifier (Lean 4 + Mathlib)

Để GPU 96GB chỉ tập trung tính toán tensor, Lean 4 compiler được cách ly trên CPU:

#### Cách A: Chạy bằng Docker (Khuyến nghị)
```bash
cd docker
docker-compose up -d --build
```
Container được gán cứng tài nguyên:
- Giới hạn CPU: `32 vCPU` (16-32 threads)
- Giới hạn RAM: `32GB`
- Chế độ mạng: `network_mode: none` (cách ly bảo mật tuyệt đối)

#### Cách B: Chạy trực tiếp trên Host Linux
```bash
./scripts/setup_lean4.sh
```

---

## 4. Stage 1: Cold-Start SFT (Định hình tư duy & Cú pháp)

Dạy mô hình phong cách suy luận tactic, cấu trúc thẻ suy luận `<think>...</think>`, và cú pháp chứng minh Mathlib chuẩn mực.

```bash
./scripts/run_sft.sh configs/sft_config.yaml outputs/qwen3.8-27b-lean4-sft
```

* File output LoRA adapter hoàn chỉnh sẽ được lưu tại: `outputs/qwen3.8-27b-lean4-sft/final`.

---

## 5. Stage 2: RLVR với GRPO (Tiệm cận Frontier)

Sử dụng Lean 4 compiler làm Ground-Truth Verifier để loại bỏ ảo giác và tối ưu hóa xác suất tìm được đường chứng minh hợp lệ.

### Chiến lược Quản lý VRAM Rollout & Train (Colocate Time-Sharing):
* Khi **Rollout**: Khởi tạo engine vLLM với `gpu_memory_utilization: 0.45` trên cùng 1 GPU 96GB (dùng chung base weights BF16 để sample nhanh $K=4$ trajectories song song).
* Khi **Backward**: vLLM giải phóng bộ nhớ đệm; PyTorch forward/backward trên adapter LoRA.
* **KL Penalty ($\beta$):** `0.02` (giữ policy không trôi quá xa khỏi base SFT).
* **Temperature ($\tau$):** `0.7`.

Khởi chạy GRPO:
```bash
./scripts/run_grpo.sh configs/grpo_config.yaml outputs/qwen3.8-27b-lean4-sft/final outputs/qwen3.8-27b-lean4-grpo
```

---

## 6. Cơ chế Reward Logic Chuẩn xác của Sandbox Verifier

Bộ kiểm tra formal verification áp dụng strict timeout `15.0s` cho mỗi bước chứng minh để tránh các tactic lặp vô tận (như `simp` loop hoặc search combinatorial bế tắc):

| Trường hợp | Reward | Mô tả kỹ thuật |
| :--- | :---: | :--- |
| **Proved (Không còn `sorry`)** | **`+1.0`** | Lean 4 compiler trả về return code 0, không còn subgoals mở, không chứa `sorry` hay `admit`. |
| **Partial Progress** | **`+0.1`** | Tactic hợp lệ, không lỗi cú pháp nhưng chưa giải quyết hết toàn bộ subgoals. |
| **Syntax / Type Error** | **`-0.5`** | Compiler báo lỗi cú pháp, sai type, hoặc tactic không tồn tại. |
| **Timeout (> 15s)** | **`-0.3`** | Tactic bị treo hoặc lặp vô tận, process bị ngắt cưỡng bức. |
| **Định dạng `<think>` và code block** | **`+0.3`** | Khuyến khích mô hình suy nghĩ cẩn thận trong `<think>` trước khi đưa code. |

Kiểm thử nhanh verifier độc lập:
```bash
python3 scripts/test_verifier.py
```
