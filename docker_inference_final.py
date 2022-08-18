"""Inference script for docker submission"""
import argparse
from pathlib import Path
from custom_nnunet_predict import predict_from_folder
from PostProcessing import postprocess_file

parser = argparse.ArgumentParser(description="Run full inference pipeline")
parser.add_argument(
    "--input_dir",
    type=str,
    metavar="",
    required=True,
    help="Path to input images",
)
parser.add_argument(
    "--output_dir",
    type=str,
    metavar="",
    required=True,
    help="Path to output",
)


def main(input_dir, output_dir):
    # Sanitize input
    output_dir = output_dir.split("\r")[0]

    # Dummy output to test if we can write to output
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Created directory {output_dir}")

    dummy_filepath = Path(output_dir) / "dummy.txt"
    dummy_filepath.touch()
    print(f"Created dummy file {dummy_filepath}")

    # Setup arguments
    args = {}

    # These are crucial for performance
    args["tta"] = False
    args["step_size"] = 0.9

    # These are specific to testing/deploy environment
    args[
        "model"
    ] = "/opt/algorithm/nnunet/results/nnUNet/3d_fullres/Task101_FLARE/nnUNetTrainerV2__nnUNetPlansv2.1/"

    # These should not be changed
    args["input_folder"] = input_dir
    args["output_folder"] = output_dir
    args["folds"] = "all"
    args["save_npz"] = False
    args["num_threads_preprocessing"] = 1
    args["num_threads_nifti_save"] = 1
    args["lowres_segmentations"] = None
    args["part_id"] = 0
    args["num_parts"] = 1
    args["mixed_precision"] = True
    args["mode"] = "fastest"
    args["overwrite_all_in_gpu"] = None
    args["checkpoint_name"] = "model_final_checkpoint"
    args["segmentation_export_kwargs"] = None
    args["disable_postprocessing"] = False
    args["use_gaussian"] = True

    predict_from_folder(**args)

    print("GPU prediction done")

    for filepath in Path(output_dir).glob("*.nii.gz"):
        print(f"Postprocessing {filepath}")
        postprocess_file(filepath)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
