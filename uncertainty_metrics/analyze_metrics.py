import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

save_path = Path("")
df = pd.read_csv("/path/to/uncertainty_metrics.csv")
metrics_dict = df.to_dict()

# Rank cases by metrics
ranking_dict = {}
for idx, metric in enumerate(metrics_dict):
    if metric == "case_number":
        continue
    values = list(metrics_dict[metric].values())
    ranking = np.argsort(values)
    ranking_dict[metric] = ranking

# For each case, find biggest drop in ranking
for case_idx in range(len(values)):
    labels = []
    positions = {}
    for label, metric in ranking_dict.items():
        position = np.where(metric == case_idx)[0][0]
        positions[label] = position
        labels.append(label)

    max_diff = -1
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            label_a = labels[i]
            label_b = labels[j]
            diff = positions[label_a] - positions[label_b]
            abs_diff = np.abs(diff)
            if abs_diff > max_diff:
                max_diff = abs_diff
                if diff > 0:
                    from_label = label_a
                    to_label = label_b
                else:
                    from_label = label_b
                    to_label = label_a
                from_score = positions[from_label]
                to_score = positions[to_label]

    # print(
    #     f"{case_idx} : {max_diff}, {from_score} -> {to_score}, {metrics_dict[from_label][case_idx]:.2f} -> {metrics_dict[to_label][case_idx]:.2f} , {from_label} -> {to_label}"
    # )
    #


def find_min_organ(case_idx, print_dsc=False):
    # Load pkl
    pkl_path = save_path / f"case_{case_idx:0>5}.pkl"
    with open(pkl_path, "rb") as pkl_file:
        pairwise_DSC_metrics = pickle.load(pkl_file)

    # Get mean DSC per organ
    mean_DSC_per_organ = {}
    for organ, dsc_list in pairwise_DSC_metrics["DSC_per_organ"].items():
        mean_DSC_per_organ[organ] = np.mean(dsc_list)
    ranked_DSC_per_organ = {
        k: v for k, v in sorted(mean_DSC_per_organ.items(), key=lambda item: item[1])
    }

    if print_dsc:
        # Pretty print
        print()
        print(f"Case {case_idx}")
        print(f"-----------")
        for organ, dsc in ranked_DSC_per_organ.items():
            print(f"{organ}: {dsc:.2f}")
        print()

    for organ in ranked_DSC_per_organ:
        return organ


threshold = 0.95
for idx, metric in enumerate(metrics_dict):
    if metric != "pair_mean_organ_min":
        continue

    # Threshold
    min_organs = {}
    for key, val in metrics_dict[metric].items():
        if val < threshold:
            min_organ = find_min_organ(metrics_dict["case_number"][key], print_dsc=True)
            if min_organ not in min_organs:
                min_organs[min_organ] = 0

            min_organs[min_organ] += 1

plt.figure()
plt.bar(list(min_organs.keys()), list(min_organs.values()))
plt.title(
    f"Worst organs after thresholding at {threshold} (n={np.sum(list(min_organs.values()))})"
)
plt.show()

plt.figure()
for idx, metric in enumerate(metrics_dict):
    if metric == "case_number":
        continue
    plt.subplot(1, 4, idx)
    plt.title(metric)
    plt.hist(metrics_dict[metric].values(), bins=50)
plt.show()

# Plot pair_mean_organ_min with threshold
plt.figure()
for idx, metric in enumerate(metrics_dict):
    if metric != "pair_mean_organ_mean":
        continue

    # Threshold
    after_threshold = []
    for val in metrics_dict[metric].values():
        if val > threshold:
            after_threshold.append(val)

    plt.title(
        f"{metric}, {len(after_threshold)} cases after thresholding at {threshold}"
    )
    plt.vlines(threshold, 0, 300, colors=["black"])
    plt.hist(metrics_dict[metric].values(), bins=100)
plt.show()
