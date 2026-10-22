import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.lines import Line2D

# validated reference palette (slots 1-3): body = blue, hand = orange, object = aqua
BODY, HAND, OBJ = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"

# Dataset averages (computed 2026-10-04 from the bench source data):
#   T-pose height over all 762 sequences of OMOMO (602, 1.73 m) + HUMOTO (100, 1.55 m) + HiPHI (60, 1.72 m) -> 1.705 m
#   human middle-finger chain (wrist->MCP->PIP->DIP->tip) over the 324 hands of HUMOTO (18.2 cm) + HiPHI (17.9 cm) -> 18.1 cm
#   Wuji Hand 2 chain (flange->middle tip, hand_bench/hand_model.py HandModel.size) -> 20.9 cm
H_h, H_r = 1.705, 1.32   # average human / robot standing height (m)
L_h, L_r = 18.1, 20.9    # average human / Wuji middle-finger chain length (cm)
D = 8.0                  # mug diameter in the demonstration (cm)
s = H_r / H_h            # UMR body scale: 0.77
ratio = L_r / L_h        # AnyTeleop hand ratio: 1.15

worlds = [
    ("Human demonstration (dataset average)", H_h, L_h, D, "hand / object = 1.00  (reference)",
     "Everything at its real size."),
    (f"Robot world, objects scaled by s = {s:.2f}\n(UMR convention, current pipeline)", H_r, L_r, D * s, f"hand / object = {ratio / s:.2f}  (+{ratio / s - 1:.0%} vs human)",
     f"Body-level contacts match (chairs, boxes, reach),\nbut the hand is {ratio / s - 1:.0%} too big for the object:\nfingers must close tighter than the human did."),
]

fig = plt.figure(figsize=(10, 8.4), facecolor=SURF)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.05, wspace=0.12, left=0.05, right=0.97, top=0.845, bottom=0.13)
fig.suptitle("One scale factor cannot serve the body and the hand", x=0.05, ha="left", fontsize=17, color=INK, weight="bold", y=0.98)
fig.text(0.05, 0.945, f"Robot body is {1-s:.0%} shorter than the average human (s = {H_r} / {H_h:.2f} = {s:.2f}),\nbut Wuji Hand 2 is {ratio-1:.0%} longer than the average human hand (ratio = {L_r} / {L_h} = {ratio:.2f}).",
         fontsize=10, color=INK2, va="top", linespacing=1.3)
fig.text(0.05, 0.880, "Human averages: T-pose height over OMOMO + HUMOTO + HiPHI (762 sequences); middle-finger chain over HUMOTO + HiPHI (324 hands).",
         fontsize=8.5, color=INK2)

for c, (title, Hb, Lhand, Dobj, ratio_txt, note) in enumerate(worlds):
    # ---- top row: standing height (metres), to a shared scale
    ax = fig.add_subplot(gs[0, c]); ax.set_facecolor(SURF)
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.25)
    ax.add_patch(FancyBboxPatch((4.1, 0), 1.8, Hb, boxstyle="round,pad=0,rounding_size=0.06", color=BODY, lw=0))
    ax.hlines(0, 0, 10, color=INK2, lw=1)
    ax.text(5.0, Hb + 0.06, f"{Hb:.2f} m", ha="center", va="bottom", fontsize=11, color=INK)
    ax.text(5.0, 2.22, title, ha="center", va="top", fontsize=11.5, color=INK, weight="bold")
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    if c == 0:
        ax.text(0.2, 0.75, "standing\nheight", fontsize=9.5, color=INK2, va="center")
    # ---- bottom row: hand length vs object width (centimetres), shared scale
    ax = fig.add_subplot(gs[1, c]); ax.set_facecolor(SURF)
    ax.set_xlim(0, 40); ax.set_ylim(-2, 18); ax.set_aspect("equal")
    ax.add_patch(FancyBboxPatch((2, 10.2), Lhand, 2.6, boxstyle="round,pad=0,rounding_size=1.0", color=HAND, lw=0))
    ax.text(2 + Lhand / 2, 15.6, f"hand {Lhand:.1f} cm", ha="center", fontsize=10.5, color=INK)
    cx = 2 + Lhand + 4 + Dobj / 2
    ax.add_patch(Circle((cx, 11.5), Dobj / 2, color=OBJ, lw=0))
    ax.text(cx, 15.6, f"mug Ø {Dobj:.1f} cm", ha="center", fontsize=10.5, color=INK)
    ax.text(20, 4.6, ratio_txt, ha="center", fontsize=12.5, color=INK, weight="bold")
    ax.text(20, 3.0, note, ha="center", va="top", fontsize=9.5, color=INK2, linespacing=1.35)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    if c == 0:
        ax.text(0.3, 17.4, "hand vs object (top view), same scale", fontsize=9.5, color=INK2)

# legend by entity (color follows the entity across both worlds)
handles = [Line2D([0], [0], marker="s", ms=10, lw=0, color=BODY, label="body"),
           Line2D([0], [0], marker="s", ms=10, lw=0, color=HAND, label="hand (middle-finger chain)"),
           Line2D([0], [0], marker="o", ms=10, lw=0, color=OBJ, label="hand-held object")]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.02), ncol=3, frameon=False, fontsize=10)
fig.text(0.97, 0.03, "Body contacts want objects × s; grasps want objects × ratio.\nNo single object scale satisfies both.",
         ha="right", va="bottom", fontsize=9.5, color=INK2, linespacing=1.4)
out = "/home/kaizhao/projects/retarget/omomo_bench/results/figures/scaling_conflict.png"
fig.savefig(out, dpi=160, facecolor=SURF)
print("wrote", out)
