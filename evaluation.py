"""This file has function for calculating DSC and NSD in the identical
way to FLARE evaluation, but a bit easier to use."""
import numpy as np
import argparse
from collections import OrderedDict
from flare_evaluation_zip.SurfaceDice import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient,
)
from pathlib import Path
import nibabel as nb
import pandas as pd
from tqdm import tqdm

LABEL_TOLERANCE = OrderedDict(
    {
        "Liver": 5,
        "RK": 3,
        "Spleen": 3,
        "Pancreas": 5,
        "Aorta": 2,
        "IVC": 2,
        "RAG": 2,
        "LAG": 2,
        "Gallbladder": 2,
        "Esophagus": 3,
        "Stomach": 5,
        "Duodenum": 7,
        "LK": 3,
    }
)


def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.

    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int

    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) == 1, print("mask label error!")
    z_index = np.where(organ_mask > 0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)

    return z_lower, z_upper


def get_NSD(pred, label, spacing, tolerance, only_labeled_slices=False):
    """Calculate normalized surface Dice

    Parameters
    ----------
    pred : array
        Binary mask
    label : array
        Binary mask
    spacing : tuple
    tolerance : int
    only_labeled_slices : bool
        If True, evaluate metric only on the non-zero label slices
    """
    # Handle empty label
    if np.sum(pred) == 0 and np.sum(label) == 0:
        return 1
    elif np.sum(label) == 0 and np.sum(pred) > 0:
        return 0

    # Remove zero label slices if needed
    if only_labeled_slices:
        z_lower, z_upper = find_lower_upper_zbound(label)
        label, pred = label[:, :, z_lower:z_upper], pred[:, :, z_lower:z_upper]

    surface_distances = compute_surface_distances(label, pred, spacing)
    NSD = compute_surface_dice_at_tolerance(surface_distances, tolerance)

    return NSD


def get_DSC(pred, label, only_labeled_slices=False):
    """Calculate Dice

    Parameters
    ----------
    pred : array
        Binary mask
    label : array
        Binary mask
    only_labeled_slices : bool
        If True, evaluate metric only on the non-zero label slices
    """
    # Handle empty label
    if np.sum(pred) == 0 and np.sum(label) == 0:
        return 1
    elif np.sum(label) == 0 and np.sum(pred) > 0:
        return 0

    # Remove zero label slices if needed
    if only_labeled_slices:
        z_lower, z_upper = find_lower_upper_zbound(label)
        label, pred = label[:, :, z_lower:z_upper], pred[:, :, z_lower:z_upper]

    DSC = compute_dice_coefficient(label, pred)

    return DSC


def get_metrics_per_organ(pred, label, spacing, only_DSC=False):
    metrics = {}
    metrics["DSC"] = {}
    metrics["NSD"] = {}
    for i, organ in enumerate(LABEL_TOLERANCE.keys(), 1):
        prediction = pred == i
        label_binary = label == i

        # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
        if i == 5 or i == 6 or i == 10:
            only_labeled_slices = True
        else:
            only_labeled_slices = False

        metrics["DSC"][organ] = get_DSC(
            prediction, label_binary, only_labeled_slices=only_labeled_slices
        )
        if not only_DSC:
            metrics["NSD"][organ] = get_NSD(
                prediction,
                label_binary,
                spacing,
                LABEL_TOLERANCE[organ],
                only_labeled_slices=only_labeled_slices,
            )

    return metrics


def evaluate_set(pred_path, label_path, results_csv_filepath=None):
    pred_path = Path(pred_path)
    label_path = Path(label_path)

    prediction_filepaths = list(pred_path.rglob("*.nii.gz"))
    prediction_filepaths.sort()

    # Prepare metric dict
    seg_metrics = OrderedDict()
    seg_metrics["Name"] = list()
    for organ in LABEL_TOLERANCE.keys():
        seg_metrics["{}_DSC".format(organ)] = list()
    for organ in LABEL_TOLERANCE.keys():
        seg_metrics["{}_NSD".format(organ)] = list()

    # Calculate metrics for each prediction
    for prediction_filepath in tqdm(
        prediction_filepaths, desc="Evaluating DSC and NSD"
    ):
        name = prediction_filepath.name

        # Load nifti
        label_nii = nb.load(label_path / name)
        spacing = label_nii.header.get_zooms()
        label = np.uint8(label_nii.get_fdata())
        prediction = np.uint8(nb.load(prediction_filepath).get_fdata())

        # Get metrics
        metrics = get_metrics_per_organ(prediction, label, spacing)

        # Save in dict
        seg_metrics["Name"].append(name)
        for i, organ in enumerate(LABEL_TOLERANCE.keys(), 1):
            seg_metrics["{}_DSC".format(organ)].append(round(metrics["DSC"][organ], 4))
        for i, organ in enumerate(LABEL_TOLERANCE.keys(), 1):
            seg_metrics["{}_NSD".format(organ)].append(round(metrics["NSD"][organ], 4))

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(results_csv_filepath, index=False)


def get_mean_metrics(metrics):
    """Get mean DSC and NSD from metrics dataframe"""
    DSC = []
    NSD = []
    for idx, row in metrics.iterrows():
        row_dict = dict(row)
        for label, value in row_dict.items():
            if "DSC" in label:
                DSC.append(value)
            if "NSD" in label:
                NSD.append(value)

    return np.mean(DSC), np.mean(NSD)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()

    base_pred_path = Path("")
    pred_path = base_pred_path / args.folder
    pred_path = pred_path / "inference_full_res_results"
    label_path = ""
    results_csv_filepath = pred_path / "metrics.csv"

    if results_csv_filepath.is_file():
        print("Loading metrics from metrics.csv")
    else:
        evaluate_set(pred_path, label_path, results_csv_filepath=results_csv_filepath)

    metrics = pd.read_csv(results_csv_filepath)

    DSC, NSD = get_mean_metrics(metrics)
    print(f"Mean DSC: {DSC*100}")
    print(f"Mean NSD: {NSD*100}")
