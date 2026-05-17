import torch, numpy as np, matplotlib
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
BG_COLOR  = (127, 127, 127)

# Tách thành nhiều prompt đơn — SAM3 match tốt hơn từng concept riêng
PROMPTS = [
    "robot arm",
    "robotic hand",
    "gripper",
    "mechanical finger",
]


class SAM3Segmenter:
    def __init__(self, model_id: str = "facebook/sam3"):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model     = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def _segment_one(self, image: Image.Image,
                       prompt: str, threshold: float) -> torch.Tensor | None:
        """Một lần gọi model với một prompt đơn.
        Trả về tensor bool [N,H,W] hoặc None nếu không có gì."""
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        masks = result["masks"]
        return masks if len(masks) > 0 else None

    def segment_multi_prompt(self, image_path: str | Path,
                              prompts: list[str],
                              threshold: float = 0.3) -> np.ndarray | None:
        """
        Gọi model N lần với N prompt riêng biệt.
        Union tất cả masks → trả về alpha mask H×W (uint8 0/255).
        threshold thấp (0.3) để bắt detection yếu trên real-world.
        """
        image    = Image.open(image_path).convert("RGB")
        h, w     = image.size[1], image.size[0]
        combined = np.zeros((h, w), dtype=bool)

        for prompt in prompts:
            masks = self._segment_one(image, prompt, threshold)
            if masks is not None:
                # union từng mask vào combined
                combined |= masks.any(dim=0).cpu().numpy()

        if not combined.any():
            return None  # không tìm thấy gì với mọi prompt

        return combined.astype(np.uint8) * 255

    def remove_background(self, image_path: str | Path,
                           alpha: np.ndarray,
                           bg_color: tuple = BG_COLOR) -> Image.Image:
        image      = Image.open(image_path).convert("RGB")
        background = Image.new("RGB", image.size, bg_color)
        background.paste(image, mask=Image.fromarray(alpha, mode="L"))
        return background


class BatchProcessor:
    def __init__(self, root: str | Path, prompts: list[str],
                 threshold: float = 0.3):
        self.root      = Path(root)
        self.prompts   = prompts
        self.threshold = threshold
        self.out_root  = self.root.parent / "image_no_bg"
        self.seg       = SAM3Segmenter()

    def _out_path(self, img_path: Path) -> Path:
        out = self.out_root / img_path.relative_to(self.root)
        out.parent.mkdir(parents=True, exist_ok=True)
        return out.with_suffix(".png")

    def run(self, splits: list[str] = ["high", "low"]):
        for split in splits:
            images = sorted((self.root / split).rglob("*"))
            images = [p for p in images if p.suffix.lower() in IMG_EXTS]
            print(f"\n[{split}] {len(images)} ảnh")

            missed = 0
            for img_path in tqdm(images, desc=split):
                try:
                    alpha = self.seg.segment_multi_prompt(
                        img_path, self.prompts, self.threshold
                    )
                    if alpha is None:
                        missed += 1
                        print(f"  ✗ no mask: {img_path.name}")
                        continue
                    out = self.seg.remove_background(img_path, alpha)
                    out.save(self._out_path(img_path))
                except Exception as e:
                    print(f"  ✗ error {img_path.name}: {e}")

            print(f"  → missed {missed}/{len(images)} frames")
        print(f"\n✓ Xong! Output: {self.out_root}")


if __name__ == "__main__":
    # ── Chạy trực tiếp: python sam3.py ────────────────────────────
    ROOT = "/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/cosmos_synthetic_data_image"
    processor = BatchProcessor(ROOT, prompts=PROMPTS, threshold=0.3)
    processor.run()