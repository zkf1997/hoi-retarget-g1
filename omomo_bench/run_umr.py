#!/usr/bin/env python3
"""Run UMR (HSI/HOI pipeline) on prepared OMOMO sequences targeting the Unitree G1.

Run in the `umr` conda env:
    python run_umr.py --hand none --workers 3                    # body-only G1, UMR-native scaling  -> results/umr_none
    python run_umr.py --hand brainco --scaling omni --workers 3  # G1 + BrainCo (hands at rest), OmniRetarget-style scaling
    python run_umr.py --seqs sub16_largebox_000 --hand wuji

--hand    none | brainco | wuji    robot.xml = robots/g1_29dof_<hand>.xml (make_hand_mjcf.py); hand joints pinned at
                                   hand_rest_pose.json through joint_limits, so only the 29 body DoF are optimised
--scaling umr | omni | real        umr  = native: object mesh and xyz x smpl_scale, no ground alignment (data/umr)
                                   real = object mesh at REAL size, xyz x smpl_scale (UMR retarget_object_size=original,
                                          same data/umr inputs; the body adapts to the real-size object) -> umr_<hand>_real
                                   realhand = real-size mesh AND the robot-side object trajectory anchored to the human wrist
                                          with the wrist->object offset at real scale (data/umr_realhand from
                                          prepare_realsize_hand_anchor.py; retarget_prop_<obj>.csv) -> umr_<hand>_realhand
                                   omni = OmniRetarget-style: real-size object (retarget_object_size=original), human
                                          ground-aligned, object z anchored at frame-0 height (data/umr_omniscale, see
                                          prepare_scaling_variants.py)
Config = configs/umr_g1_<hand>[_omniscale].json. Per subject (one SMPL-X template) the first sequence trains the
point-cloud correspondence (~30 s); all others reuse the slots. Correspondence depends on the robot only, so both
scaling variants share it.
Set BENCH_DATASET=humoto to use data_humoto/ and results_humoto/.
Results: results/<tag>/<seq>.npz (tag default umr_<hand>[_omniscale]); qpos [T, 7+ndof] in the robot MJCF joint order
(`robot_joint_names`), fps 30, smpl_scale, ground_z; extra keys hand / scaling / object_mesh_scale added here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UMR = HERE.parent / "UMR"
# Dataset switch: BENCH_DATASET=omomo (default) | humoto  ->  data[_humoto]/, results[_humoto]/, omni_workdir[_humoto]/
DATASET = os.environ.get("BENCH_DATASET", "omomo")
SUFFIX = "" if DATASET == "omomo" else f"_{DATASET}"
DATA = HERE / f"data{SUFFIX}"
RESULTS = HERE / f"results{SUFFIX}"
CONFIGS = HERE / "configs"
PY = Path(os.environ.get("UMR_PYTHON", "/home/kaizhao/software/miniconda3/envs/umr/bin/python"))


class Setup:
    def __init__(self, hand: str, scaling: str, tag: str | None, variant: str = ""):
        self.hand, self.scaling = hand, scaling
        sfx = {"umr": "", "omni": "_omniscale", "real": "_real", "realhand": "_realhand", "multi": "_multi"}[scaling]
        self.config = CONFIGS / f"umr_g1_{hand}{'_real' if scaling == 'realhand' else sfx}{('_' + variant) if variant else ''}.json"
        self.robot_name = json.loads(self.config.read_text())["robot"]["name"]
        self.data_dir = DATA / {"omni": "umr_omniscale", "realhand": "umr_realhand", "multi": "umr_multi"}.get(scaling, "umr")
        self.tag = tag or f"umr_{hand}" + sfx + (("_" + variant) if variant else "")
        self.results = RESULTS / self.tag

    def subject_config(self, subject: str, slots: Path | None) -> Path:
        cfg = json.loads(self.config.read_text())
        if slots is not None:
            cfg.setdefault("correspondence", {})["slots"] = str(slots)
        p = CONFIGS / "generated" / f"{self.config.stem}_{subject}{'' if slots is None else '_slots'}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cfg, indent=1))
        return p

    def slots_path(self, seed_seq: str) -> Path:
        return UMR / "output" / f"correspondence_{self.robot_name}_{seed_seq}_hsi_hoi" / "correspondence_slots_final.npz"

    def existing_slots(self, subject: str) -> str | None:
        prev = sorted((UMR / "output").glob(f"correspondence_{self.robot_name}_{subject}_*_hsi_hoi/correspondence_slots_final.npz"))
        return prev[0].parent.name[len(f"correspondence_{self.robot_name}_"):-len("_hsi_hoi")] if prev else None


def annotate(npz: Path, **extra) -> None:
    d = dict(np.load(npz, allow_pickle=True))
    d.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez(npz, **d)


def run_one(st: Setup, seq: str, config: Path, stage: str = "all", extra: list[str] | None = None) -> dict:
    st.results.mkdir(parents=True, exist_ok=True)
    logs = st.results / "logs"; logs.mkdir(exist_ok=True)
    out_npz = st.results / f"{seq}.npz"
    cmd = [str(PY), str(UMR / "scripts" / "humanoid_retarget_pipeline_hsi_hoi.py"),
           "--config", str(config), "--defaults", str(UMR / "humanoid_retarget_defaults_hsi_hoi_standard.json"),
           "--data", str(st.data_dir), "--seq-key", seq, "--out", str(out_npz), "--stage", stage, "--skip-view"]
    cmd += extra or []
    t0 = time.time()
    with open(logs / f"{seq}.log", "w") as lf:
        lf.write(" ".join(cmd) + "\n\n"); lf.flush()
        rc = subprocess.call(cmd, cwd=UMR, stdout=lf, stderr=subprocess.STDOUT, env={**os.environ, "MUJOCO_GL": "egl"})
    dt = time.time() - t0
    ok = rc == 0 and out_npz.exists()
    if ok:
        s = float(np.asarray(np.load(out_npz)["smpl_scale"]).reshape(-1)[0])
        extra = {}
        rp = sorted((st.data_dir / "test" / seq).glob("retarget_prop_*.csv"))
        if rp:  # robot-world object trajectory differs from source x s: store it for the hand stages / evaluation / viewers
            rows = np.loadtxt(rp[0], delimiter=",", skiprows=1)
            extra["obj_pos_robot"] = (rows[:, :3] * s).astype(np.float32)
        # objects are scaled with the human (x s) in every variant except the real-size ones
        annotate(out_npz, hand=st.hand, scaling=st.scaling, object_mesh_scale=(1.0 if st.scaling in ("real", "realhand", "omni") else s), **extra)
    return {"seq": seq, "ok": ok, "rc": rc, "seconds": round(dt, 1), "out": str(out_npz) if ok else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", choices=["none", "brainco", "wuji"], default="none")
    ap.add_argument("--scaling", choices=["umr", "omni", "real", "realhand", "multi"], default="umr",
                    help="... multi = every object of the sequence (hand-held + furniture, data*/umr_multi from prepare_umr_multi.py, solver.object_multi)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--index", type=Path, default=DATA / "seqs_test.json")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-slot-reuse", action="store_true", help="retrain correspondence for every sequence (UMR default)")
    ap.add_argument("--variant", default="", help="config variant suffix: configs/umr_g1_<hand>_<variant>.json -> results/umr_<hand>_<variant> (e.g. v2 = heavier hand/forearm surface costs)")
    ap.add_argument("extra", nargs="*", help="extra args passed to the UMR pipeline")
    args = ap.parse_args()
    st = Setup(args.hand, args.scaling, args.tag, args.variant)
    if not st.data_dir.exists():
        raise SystemExit(f"{st.data_dir} missing (run prepare_omomo.py / prepare_scaling_variants.py)")

    index = json.loads(args.index.read_text())
    if args.seqs:
        index = [e for e in index if e["seq"] in set(args.seqs)]
    todo = [e for e in index if args.force or not (st.results / f"{e['seq']}.npz").exists()]
    print(f"[{st.tag}] {len(todo)} sequences to run (config {st.config.name}, data {st.data_dir.name})")
    summary = []

    # 1) one seed sequence per subject trains the correspondence (sequentially, GPU)
    seeds: dict[str, str] = {}
    if not args.no_slot_reuse:
        for e in todo:
            seeds.setdefault(e["subject"], e["seq"])
        for sub, seq in list(seeds.items()):
            prev = st.existing_slots(sub)
            if prev and not args.force:
                seeds[sub] = prev
                print(f"[seed] {sub}: reusing slots from {prev}"); continue
            r = run_one(st, seq, st.subject_config(sub, None), stage=args.stage, extra=args.extra)
            r["role"] = "seed"; summary.append(r); print(r, flush=True)
            if not r["ok"]:
                print(f"[seed] {sub} FAILED, see {st.results}/logs/{seq}.log"); return

    # 2) everything else, in parallel, reusing slots
    done = {r["seq"] for r in summary}
    rest = [e for e in todo if e["seq"] not in done]
    def cfg_for(e):
        return st.subject_config(e["subject"], None if args.no_slot_reuse else st.slots_path(seeds[e["subject"]]))
    with ThreadPoolExecutor(max(1, args.workers)) as ex:
        futs = {ex.submit(run_one, st, e["seq"], cfg_for(e), args.stage, args.extra): e for e in rest}
        for f in as_completed(futs):
            r = f.result(); summary.append(r); print(r, flush=True)
    st.results.mkdir(parents=True, exist_ok=True)
    sp = st.results / "run_summary.json"
    prev = {r["seq"]: r for r in json.loads(sp.read_text())} if sp.exists() else {}
    prev.update({r["seq"]: r for r in summary})
    sp.write_text(json.dumps(list(prev.values()), indent=1))
    print(f"done: {sum(r['ok'] for r in summary)}/{len(summary)} ok")


if __name__ == "__main__":
    sys.exit(main())
