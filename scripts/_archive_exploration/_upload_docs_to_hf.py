"""Upload technical docs (WARPDYN_METHOD.md + DOANH_EVAL175.md) to HF."""
from pathlib import Path
from huggingface_hub import HfApi

REPO = Path("/mnt/data/sftp/data/quangpt3/gcvwm/calibration/feepe/feature_matching_eval_hallucination")
HF_REPO = "twanghcmut/wmbench"
PREFIX = "doanh_eval175"

token = (Path.home() / ".cache/huggingface/token").read_text().strip()
api = HfApi(token=token)

for fn in ["WARPDYN_METHOD.md", "DOANH_EVAL175.md"]:
    src = REPO / fn
    api.upload_file(path_or_fileobj=str(src),
                    path_in_repo=f"{PREFIX}/{fn}",
                    repo_id=HF_REPO, repo_type="dataset")
    print(f"→ {PREFIX}/{fn}")

print(f"\nHF: https://huggingface.co/datasets/{HF_REPO}/tree/main/{PREFIX}")
