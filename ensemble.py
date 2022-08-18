"""Script that ensembles predictions for 5 folds."""
from mpire import WorkerPool
from utils import ensemble_predictions, load_softmax
from pathlib import Path
import SimpleITK as sitk
import numpy as np


def ensemble_case(case_idx, prediction_dir, case_dir, save_dir):
    save_path = save_dir / f"Case_{case_idx:0>5}.nii.gz"

    # Check if case already exists
    if save_path.is_file():
        print(f"{save_path.name} already exists, skipping...")
        return

    # Load npz files
    predictions = []
    for fold_idx in range(5):
        softmax_path = prediction_dir / f"fold_{fold_idx}" / f"Case_{case_idx:0>5}.npz"
        prediction = load_softmax(softmax_path)
        predictions.append(prediction)

    # prediction = np.random.random(prediction[0].shape)
    prediction = ensemble_predictions(predictions)

    # Load case nifti
    case_path = case_dir / f"Case_{case_idx:0>5}_0000.nii.gz"
    case_nifti = sitk.ReadImage(str(case_path))

    # Convert prediction to nifti
    prediction_sitk = sitk.GetImageFromArray(prediction.astype(np.uint8))
    prediction_sitk.CopyInformation(case_nifti)

    # Save in appropriate place
    sitk.WriteImage(prediction_sitk, str(save_path))

    print(f"Finished ensembling {save_path.name}")


if __name__ == "__main__":
    prediction_dir = Path("")
    case_dir = Path("")
    save_dir = Path("")

    # Assemble inputs for multithreading
    inputs = []
    for i in range(2000):
        single_input = {
            "case_idx": i + 1,
            "prediction_dir": prediction_dir,
            "case_dir": case_dir,
            "save_dir": save_dir,
        }
        inputs.append(single_input)

    with WorkerPool(n_jobs=6) as pool:
        results = pool.map(ensemble_case, inputs)
