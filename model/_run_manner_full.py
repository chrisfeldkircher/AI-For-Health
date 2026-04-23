"""Full manner-label extraction over train + devel. Run from model/ dir."""
import sys, time, torch
from pathlib import Path

if str(Path.cwd().resolve()) not in sys.path:
    sys.path.insert(0, str(Path.cwd().resolve()))

from features.manner import extract_manner_labels
from data.data import AudioDataset

DATA_DIR   = "../dataset/ComParE2017_Cold_4students"
CACHE_ROOT = "../cache"
BACKBONE   = "wavlm-large"
CLIP_SECS  = 8.0

t0 = time.time()
for split in ("train", "devel"):
    ds = AudioDataset(
        data_dir=DATA_DIR, split=split,
        use_mel=False, use_opensmile=False,
        pad_or_truncate_secs=CLIP_SECS,
    )
    print(f"manner[{split}] n={len(ds)}", flush=True)
    report = extract_manner_labels(
        dataset=ds, cache_root=CACHE_ROOT,
        backbone_id=f"microsoft_{BACKBONE}",
        skip_existing=True,
    )
    print(f"  wrote {report['n_written']} new labels (elapsed {time.time()-t0:.1f}s)", flush=True)

print(f"DONE total={time.time()-t0:.1f}s", flush=True)
