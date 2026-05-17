# Legacy — Pre-refactor code

Code từ pipeline 4-stage ban đầu, đã được replaced bởi `warp_score/` package.

| File | Mô tả |
|------|-------|
| `main.py` | Entry point cũ (4-stage pipeline) |
| `config.py` | Config dataclass cũ |
| `dataset.py` | Dataset loader cũ |
| `detector.py` | Detector cũ |
| `stage1_extract.py` | Feature extraction → HDF5 |
| `stage2_coreset.py` | Coreset selection (k-means) |
| `stage3_reference.py` | Reference pool selection |
| `stage4_graph.py` | Graph-based ranking |
| `roma_utils.py` | RoMaV2 utilities |
| `sam3.py` | SAM3 background removal helper |
| `warp_variance_vis.py` | Visualization script |
| `files.zip` | Archive cũ |

Dùng `python -m warp_score` thay thế.
