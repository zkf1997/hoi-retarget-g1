#!/usr/bin/env python3
"""HiPHI: per-sequence body heights for OmniRetarget (UMR scales each sequence by its own SMPL-X shape; holosoma looks up
one height per `subject` in data_hiphi/omni/height_dict.pkl).  Writes data_hiphi/seqs_test_perseq.json (subject = seq) and
adds one height_dict entry per sequence (the original single 'hiphi' entry is kept; backup height_dict_single.pkl).
    python scripts/make_hiphi_perseq_index.py
Then: BENCH_DATASET=hiphi python run_omni.py --hand wuji --scaling umr --index data_hiphi/seqs_test_perseq.json ...
"""
import json, pickle, shutil
import numpy as np
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data_hiphi"
hp = D / "omni" / "height_dict.pkl"
if not (D / "omni" / "height_dict_single.pkl").exists():
    shutil.copy(hp, D / "omni" / "height_dict_single.pkl")
h = pickle.load(open(hp, "rb")); idx = json.load(open(D / "seqs_test.json")); out = []
for e in idx:
    s = np.load(D / "source" / f"{e['seq']}.npz", allow_pickle=True)
    h[e["seq"]] = float(s["tpose_height"]); out.append(dict(e, subject=e["seq"]))
pickle.dump(h, open(hp, "wb")); json.dump(out, open(D / "seqs_test_perseq.json", "w"), indent=1)
print(f"{len(out)} sequences, heights {min(h[e['seq']] for e in idx):.3f}-{max(h[e['seq']] for e in idx):.3f} m")
