"""Uncertainty metrics for predictions"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation import get_metrics_per_organ
import numpy as np
from mpire import WorkerPool


def get_metrics_wrapper(label_a, label_b):
    return get_metrics_per_organ(
        label_a,
        label_b,
        spacing=(1, 1, 1),
        only_DSC=True,
    )


def pairwise_DSC(labels):
    """Compute a pairwise DSC metric for multiple predictions

    Parameters
    ----------
    labels : list
        List of numpy arrays
    """
    n_labels = len(labels)
    DSC_per_organ = None

    # Assemble inputs for multithreading
    inputs = []
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            label_a = labels[i]
            label_b = labels[j]
            single_input = {"label_a": label_a, "label_b": label_b}
            inputs.append(single_input)
    with WorkerPool(n_jobs=10) as pool:
        results = pool.map(get_metrics_wrapper, inputs)

    # Add pairwise metric to dict
    for metrics_per_organ in results:
        if DSC_per_organ is None:
            DSC_per_organ = metrics_per_organ["DSC"]
            for organ in DSC_per_organ:
                DSC_per_organ[organ] = [DSC_per_organ[organ]]
        else:
            for organ in DSC_per_organ:
                DSC_per_organ[organ].append(metrics_per_organ["DSC"][organ])

    # Save raw results
    pairwise_DSC = {}
    pairwise_DSC["DSC_per_organ"] = DSC_per_organ

    # Reduce DSC per organ
    mean_DSC_per_organ = []
    min_DSC_per_organ = []
    for organ in DSC_per_organ:
        mean_DSC_per_organ.append(np.mean(DSC_per_organ[organ]))
        min_DSC_per_organ.append(np.min(DSC_per_organ[organ]))

    # Reduce over organs
    pairwise_DSC["pair_mean_organ_mean"] = np.mean(mean_DSC_per_organ)
    pairwise_DSC["pair_min_organ_mean"] = np.mean(min_DSC_per_organ)
    pairwise_DSC["pair_mean_organ_min"] = np.min(mean_DSC_per_organ)
    pairwise_DSC["pair_min_organ_min"] = np.min(min_DSC_per_organ)

    return pairwise_DSC


def get_softmax_metrics(softmax_volumes):
    """Softmax based uncertainty metrics"""
    metrics = {}
    # Stack volumes, shape becomes (5, 14, z, y, x)
    stacked_softmax = np.stack(softmax_volumes)

    # Predictive entropy

    # Average probability per class over all 5 models
    predictive_entropy = np.mean(stacked_softmax, axis=0) + 1e-6

    # Calculate entropy per pixel
    predictive_entropy = -predictive_entropy * np.log(predictive_entropy)
    predictive_entropy = np.sum(predictive_entropy, axis=0)

    # Metric is mean entropy over all pixels
    metrics["predictive_entropy"] = np.mean(predictive_entropy)

    # Predictive variance
    predictive_variance = np.var(stacked_softmax, axis=0)

    # Metric is averaged over all classes and pixels
    metrics["predictive_variance"] = np.mean(predictive_variance)

    return metrics
