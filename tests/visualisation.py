import ast
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

with open(
    "test_results/bloom/qwen/level_ragas_scores_awf_20250804_220453.txt",
    "r",
    encoding="utf-8",
) as f1:
    data1 = ast.literal_eval(f1.read().strip())

with open(
    "test_results/bloom/qwen/level_ragas_scores_20250801_234640.txt",
    "r",
    encoding="utf-8",
) as f2:
    data2 = ast.literal_eval(f2.read().strip())

metrics = list(data1.keys())
scores1 = list(data1.values())
scores2 = list(data2.values())

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(
    x - width / 2, scores1, width, label="Agent-wise feedback", color="gray", hatch="//"
)
bars2 = ax.bar(
    x + width / 2,
    scores2,
    width,
    label="Retriever-only feedback",
    color="lightgray",
    hatch="\\\\",
)

ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=12)
ax.set_xlabel("Metric", fontsize=12)
ax.set_title("RAGAS Metrics Comparison", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.tick_params(axis="y", labelsize=10)

ax.grid(True, axis="y", linestyle="--", alpha=0.7)

# Move legend below plot
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=10
)

plt.tight_layout()
plt.savefig("ragas_metrics_comparison.pdf", bbox_inches="tight")  # optional for saving
plt.show()
