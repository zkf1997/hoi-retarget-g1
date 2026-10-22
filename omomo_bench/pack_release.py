#!/usr/bin/env python3
"""Pack the final retargeting results + everything the viewers need into one archive.

    python pack_release.py [--out /path/retarget_final_results_<date>] [--no-tar]

Layout (mirrors the repo, so extracting the archive over a clone of the `handover-2026-10` branch makes the viewers work):
    <out>/RESULTS_README.md                           storage format (also HANDOVER.md section "Result archive")
    <out>/omomo_bench/data[_humoto|_hiphi]/seqs_test.json, multi_objects*.json, source/<seq>.npz, objects/<obj>/...
    <out>/omomo_bench/results[_humoto|_hiphi]/pipe_final/<seq>.npz          final robot motions (real dirs, not symlinks)
    <out>/omomo_bench/results[_humoto|_hiphi]/final_metrics/*.{csv,md}      metrics of the final lanes
    <out>/urdf/g1_wuji_hand2/                                               G1 + Wuji Hand 2 URDF + meshes (viewer visuals)
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FINAL = {  # dataset suffix -> (final lane, metric files)
    "": ("pipe_handrepair_coupled", ["summary_handrepair_coupled.md", "wrist_gap_omomo_coupled.csv", "wrist_gap_omomo_coupled.log"]),
    "_humoto": ("pipe_v3_nfcfull", ["pipe_summary_nfcfull.md", "pipe_metrics_nfcfull.csv", "wrist_gap_final.csv", "wrist_gap_final.log"]),
    "_hiphi": ("pipe_v3_it12", ["pipe_summary_it12.md", "pipe_metrics_it12.csv", "wrist_gap_final.csv", "wrist_gap_final.log"]),
}
DATA_FILES = ["seqs_test.json", "seqs_test_perseq.json", "multi_objects.json", "multi_objects_handheld.json"]


def copy_tree(src: Path, dst: Path, pattern: str = "*"):
    dst.mkdir(parents=True, exist_ok=True); n = 0
    for p in sorted(src.rglob(pattern)):
        if p.is_file():
            q = dst / p.relative_to(src); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q); n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / f"retarget_final_results_{time.strftime('%Y-%m-%d')}"))
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args(); out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; remove it first")
    manifest = {}
    for sfx, (lane, metrics) in FINAL.items():
        d, r = HERE / f"data{sfx}", HERE / f"results{sfx}"
        od, orr = out / "omomo_bench" / f"data{sfx}", out / "omomo_bench" / f"results{sfx}"
        od.mkdir(parents=True)
        for f in DATA_FILES:
            if (d / f).exists(): shutil.copy2(d / f, od / f)
        n_src = copy_tree(d / "source", od / "source", "*.npz")
        n_obj = copy_tree(d / "objects", od / "objects")
        n_res = copy_tree(r / lane, orr / "pipe_final", "*.npz")
        (orr / "final_metrics").mkdir(parents=True)
        for f in metrics:
            if (r / f).exists(): shutil.copy2(r / f, orr / "final_metrics" / f)
        (orr / "pipe_final" / "LANE.txt").write_text(f"{lane}\n")
        manifest[f"data{sfx}"] = {"final_lane": lane, "sequences": n_res, "source_npz": n_src, "object_files": n_obj}
        print(f"data{sfx}: {n_res} final npz from {lane}, {n_src} source npz, {n_obj} object files", flush=True)
    copy_tree(ROOT / "urdf" / "g1_wuji_hand2", out / "urdf" / "g1_wuji_hand2")
    (out / "manifest.json").write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M"), "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(), "datasets": manifest}, indent=1))
    shutil.copy2(HERE / "RELEASE_FORMAT.md", out / "RESULTS_README.md")  # not README.md: extracting over a clone must not overwrite the repo README
    if not a.no_tar:
        tar = out.with_suffix(".tar.gz")
        print(f"tar -> {tar}", flush=True)
        subprocess.run(["tar", "-C", str(out.parent), "-czf", str(tar), out.name], check=True)
        print(f"done: {tar} ({tar.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
