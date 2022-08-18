"""Generate an nnunet dataset, from ensembled predictions.

This applies a threshold on uncertainty metrics as well.
"""
from pathlib import Path
from re import I
from shutil import copyfile
from tqdm import tqdm
import pandas as pd
from utils import Generate_nnUnet_dataset_json

# Parameters
uncertainty_threshold = 0.85
image_dir = Path("")
label_dir = Path("")
pseudolabel_dir = Path("")
task_dir = Path("")

# Load uncertainty scores
df = pd.read_csv("")

# Make task dir layout
task_image_dir = task_dir / "imagesTr"
task_label_dir = task_dir / "labelsTr"
task_image_dir.mkdir(parents=True, exist_ok=True)
task_label_dir.mkdir(parents=True, exist_ok=True)

# Copy annotated images
for image_path in tqdm(list((image_dir / "labeled").glob("*.nii.gz"))):
    case_id = image_path.name.split("_0000.nii.gz")[0]
    label_path = label_dir / (case_id + ".nii.gz")

    task_image_path = task_image_dir / image_path.name
    task_label_path = task_label_dir / label_path.name

    if not task_image_path.is_file():
        copyfile(image_path, task_image_path)
        copyfile(label_path, task_label_path)

# Copy images with pseudolabel
for label_path in tqdm(list(pseudolabel_dir.glob("*.nii.gz"))):
    case_id = label_path.name.split(".nii.gz")[0]
    image_path = (image_dir / "unlabeled") / (case_id + "_0000.nii.gz")
    task_image_path = task_image_dir / image_path.name
    task_label_path = task_label_dir / label_path.name
    print(task_image_path)

    if not task_image_path.is_file():
        copyfile(image_path, task_image_path)
        copyfile(label_path, task_label_path)

# Generate dataset json
Generate_nnUnet_dataset_json(str(task_dir))
