"""Draw docs/auc_{light,dark}.svg from the table run.py wrote to docs/results.md."""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for line in open("docs/results.md"):
    m = re.match(r"\| (.+?) \| (\d+) \| ([\d.]+) ± ([\d.]+) \| ([\d.]+) \|", line)
    if m:
        rows.append((m[1], int(m[2]), float(m[3]), float(m[4]), float(m[5])))

labels = ["Blood, n=14\n(notebook data)", "Blood, n=124\nadjusted", "Brain, n=69\nunadjusted", "Brain, n=69\nadjusted"]
THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#8c959f", bar="#0969da", age="#cf222e", grid="#d1d9e0"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#6e7681", bar="#58a6ff", age="#ff7b72", grid="#30363d"),
}
for name, t in THEMES.items():
    fig, ax = plt.subplots(figsize=(8.06, 3.2), dpi=100)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    x = np.arange(len(rows))
    ax.bar(x, [r[2] for r in rows], yerr=[r[3] for r in rows], color=t["bar"], width=0.55,
           error_kw=dict(ecolor=t["muted"], lw=1, capsize=3), label="random forest, held-out AUC (± sd over folds)")
    ax.scatter(x, [r[4] for r in rows], marker="_", s=900, color=t["age"], lw=2.2, zorder=3, label="age alone")
    ax.axhline(0.5, color=t["muted"], ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, color=t["fg"], fontsize=9)
    ax.set_ylim(0.3, 1.05); ax.set_ylabel("AUC", color=t["fg"])
    ax.tick_params(colors=t["muted"])
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(t["grid"])
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    for tx in leg.get_texts(): tx.set_color(t["fg"])
    fig.tight_layout(); fig.savefig(f"docs/auc_{name}.svg", facecolor=t["bg"]); plt.close(fig)
