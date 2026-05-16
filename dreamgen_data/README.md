# GR00T-Dreams Bulk Generator

Pipeline tự động sinh **100 video / prompt** từ dataset
[`nvidia/PhysicalAI-Robotics-GR00T-GR1`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1)
bằng Cosmos-Predict2-14B (engine của DreamGen / GR00T-Dreams).

Có 2 profile:
- **`high`** — config "tốt nhất" theo paper DreamGen (CFG=7.0, post-trained checkpoint, prompt refiner bật).
- **`hallucinate`** — config được tinh chỉnh để tăng tỉ lệ hallucination (CFG thấp, base checkpoint, tắt refiner, random seed + guidance jitter).

## Yêu cầu phần cứng

- 8× H100 / B200 / GB200 (model 14B + multi-GPU context parallel)
- ~250 GB disk cho checkpoint
- ~50 GB cho output mỗi prompt (100 video × 2 profile)

Nếu không đủ, dùng model 2B (sửa `--model_size 2B` trong `generate.py`) — quality kém hơn nhưng vẫn chạy được với 1 GPU 32 GB.

## Cấu trúc

```
gr00t-dreamgen-bulk/
├── README.md
├── requirements.txt
├── setup.sh              # B1: cài cosmos-predict2 + tải checkpoint
├── prepare_data.py       # B2: tải HF dataset + extract frame đầu của mỗi video
├── generate.py           # B3: bulk generate (load model 1 lần)
└── run_all.sh            # one-shot: gọi cả B2 và B3 cho cả 2 profile
```

## Cách chạy

```bash
# 1. Cài đặt (chỉ chạy 1 lần)
bash setup.sh

# 2. Tải dataset và extract frames
python prepare_data.py --out_dir data --max_items 100

# 3. Sinh 100 video / prompt cho profile "high" (1 prompt thôi để test)
torchrun --nproc_per_node=8 generate.py \
    --profile high \
    --num_videos 100 \
    --batch_json data/batch_input.json \
    --save_dir output/high \
    --start_idx 0 --end_idx 1   # chỉ chạy item đầu tiên

# 4. Profile hallucination (cùng prompt, config khác)
torchrun --nproc_per_node=8 generate.py \
    --profile hallucinate \
    --num_videos 100 \
    --batch_json data/batch_input.json \
    --save_dir output/hallucinate \
    --start_idx 0 --end_idx 1

# Hoặc chạy tất cả 100 prompts (mất ~1 tuần):
bash run_all.sh
```

## Time budget (8× H100, 14B model, 480p/16fps)

- ~3 phút / video → **100 video × 2 profile = ~10 giờ / prompt**
- 100 prompts × 10h = ~1000h compute (~6 tuần). Lý do nên chạy `--start_idx 0 --end_idx 1` trước để test.

## Tip giảm thời gian

- Dùng 2B model: `--model_size 2B` → ~30 giây / video.
- Dùng NATTEN sparse attention: thêm flag `--natten` (chỉ Hopper+).
- Giảm số video / prompt: `--num_videos 20` cho test.

## Hallucination filtering

Để khỏi tự lọc 200 video, sau khi gen xong dùng `examples/video2world_bestofn.py`
của cosmos-predict2 với Cosmos-Reason1-7B làm critic — nó sẽ chấm điểm 0–100.
Bạn lấy top-100 → high, bottom-100 → hallucinate, đỡ công thấy rõ.

Xem section "Rejection Sampling" trong
[inference_video2world.md](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/inference_video2world.md).
