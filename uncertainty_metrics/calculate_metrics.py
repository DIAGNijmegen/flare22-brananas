"""Script that will calculate uncertainty metrics for the entire dataset"""
from metrics import pairwise_DSC
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pickle


def load_nifti(path):
    image_sitk = sitk.ReadImage(str(path))
    image = sitk.GetArrayFromImage(image_sitk)
    return image


def load_softmax(softmax_path):
    softmax = np.load(softmax_path)
    softmax = softmax["softmax"]
    return softmax


def get_metrics_over_folds(case_idx):
    masks = []
    for fold_idx in range(5):
        nifti_path = (
            prediction_path / f"fold_{fold_idx}" / f"Case_{case_idx:0>5}.nii.gz"
        )

        # Get mask and slice
        if nifti_path.is_file():
            mask = load_nifti(nifti_path)
            masks.append(mask)

    # Calculate metrics
    pairwise_DSC_metrics = pairwise_DSC(masks)

    return pairwise_DSC_metrics


if __name__ == "__main__":
    prediction_path = Path("/path/to/predictions")
    save_path = Path("")

    csv_dict = {}
    csv_dict["case_number"] = []
    csv_dict["pair_mean_organ_mean"] = []
    csv_dict["pair_min_organ_mean"] = []
    csv_dict["pair_mean_organ_min"] = []
    csv_dict["pair_min_organ_min"] = []
    for i in tqdm(range(2000)):
        case_idx = i + 1

        pkl_path = save_path / f"case_{case_idx:0>5}.pkl"
        if pkl_path.is_file():
            print(f"Metrics for case {case_idx} already calculated")
            with open(pkl_path, "rb") as pkl_file:
                pairwise_DSC_metrics = pickle.load(pkl_file)
        else:
            print(f"Calculating metrics for case {case_idx}")
            # Get predictions from all folds
            pairwise_DSC_metrics = get_metrics_over_folds(case_idx)

            # Pickle metrics
            with open(pkl_path, "wb") as pkl_file:
                pickle.dump(pairwise_DSC_metrics, pkl_file)

        # Save in csv_dict
        csv_dict["case_number"].append(case_idx)
        for label, value in pairwise_DSC_metrics.items():
            if label == "DSC_per_organ":
                continue

            csv_dict[label].append(value)

    df = pd.DataFrame.from_dict(csv_dict)
    df.to_csv("/path/to/uncertainty_metrics.csv", index=False)
