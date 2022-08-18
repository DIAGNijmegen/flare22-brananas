"""This script contains a customized inference pipeline for nnunet,
optimized for the specific use case of the FLARE22 challeng.
"""
import os
import argparse
from copy import deepcopy
from typing import Tuple, Union, List
import time
import shutil

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from nnunet.inference.segmentation_export import (
    save_segmentation_nifti,
)
from batchgenerators.utilities.file_and_folder_operations import *
import torch
import SimpleITK as sitk
from nnunet.utilities.one_hot_encoding import to_one_hot
import gc

# Local imports
from model_restore import (
    load_model_and_checkpoint_files,
)


def preprocess(
    trainer,
    list_of_lists,
    output_files,
    segs_from_prev_stage=None,
):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)
    classes = list(range(1, trainer.num_classes))
    # assert isinstance(trainer, nnUNetTrainer)
    preprocess_fn = trainer.preprocess_patient
    transpose_forward = trainer.plans["transpose_forward"]

    outputs = []
    for i, l in enumerate(list_of_lists):
        output_file = output_files[i]
        print("preprocessing", output_file)
        d, _, dct = preprocess_fn(l)
        # print(output_file, dct)
        if segs_from_prev_stage[i] is not None:
            assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                ".nii.gz"
            ), ("segs_from_prev_stage" " must point to a " "segmentation file")
            seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
            # check to see if shapes match
            img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
            assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), (
                "image and segmentation from previous "
                "stage don't have the same pixel array "
                "shape! image: %s, seg_prev: %s" % (l[0], segs_from_prev_stage[i])
            )
            seg_prev = seg_prev.transpose(transpose_forward)
            seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
            seg_reshaped = to_one_hot(seg_reshaped, classes)
            d = np.vstack((d, seg_reshaped)).astype(np.float32)
        """There is a problem with python process communication that prevents us from communicating objects 
        larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
        communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long 
        enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
        patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
        then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
        filename or np.ndarray and will handle this automatically"""
        print(d.shape)
        if np.prod(d.shape) > (
            2e9 / 4 * 0.85
        ):  # *0.85 just to be save, 4 because float32 is 4 bytes
            print(
                "This output is too large for python process-process communication. "
                "Saving output temporarily to disk"
            )
            np.save(output_file[:-7] + ".npy", d)
            d = output_file[:-7] + ".npy"
        outputs.append((output_file, (d, dct)))
    return outputs


def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print(
        "This model expects %d input modalities for each image"
        % expected_num_modalities
    )
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert (
        len(files) > 0
    ), "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print(
        "Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
        np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)),
    )
    print(
        "If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc"
    )

    if len(remaining) > 0:
        print(
            "found %d unexpected remaining files in the folder. Here are some examples:"
            % len(remaining),
            np.random.choice(remaining, min(len(remaining), 10)),
        )

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids


def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        help="Must contain all modalities for each patient in the correct"
        " order (same as training). Files must be named "
        "CASENAME_XXXX.nii.gz where XXXX is the modality "
        "identifier (0000, 0001, etc)",
        required=True,
    )
    parser.add_argument(
        "-o", "--output_folder", required=True, help="folder for saving predictions"
    )
    parser.add_argument(
        "-m",
        "--model_output_folder",
        help="model output folder. Will automatically discover the folds "
        "that were "
        "run and use those as an ensemble",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--folds",
        nargs="+",
        default="None",
        help="folds to use for prediction. Default is None "
        "which means that folds will be detected "
        "automatically in the model output folder",
    )
    parser.add_argument(
        "-z",
        "--save_npz",
        required=False,
        action="store_true",
        help="use this if you want to ensemble"
        " these predictions with those of"
        " other models. Softmax "
        "probabilities will be saved as "
        "compresed numpy arrays in "
        "output_folder and can be merged "
        "between output_folders with "
        "merge_predictions.py",
    )
    parser.add_argument(
        "-l",
        "--lowres_segmentations",
        required=False,
        default="None",
        help="if model is the highres "
        "stage of the cascade then you need to use -l to specify where the segmentations of the "
        "corresponding lowres unet are. Here they are required to do a prediction",
    )
    parser.add_argument(
        "--part_id",
        type=int,
        required=False,
        default=0,
        help="Used to parallelize the prediction of "
        "the folder over several GPUs. If you "
        "want to use n GPUs to predict this "
        "folder you need to run this command "
        "n times with --part_id=0, ... n-1 and "
        "--num_parts=n (each with a different "
        "GPU (for example via "
        "CUDA_VISIBLE_DEVICES=X)",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        required=False,
        default=1,
        help="Used to parallelize the prediction of "
        "the folder over several GPUs. If you "
        "want to use n GPUs to predict this "
        "folder you need to run this command "
        "n times with --part_id=0, ... n-1 and "
        "--num_parts=n (each with a different "
        "GPU (via "
        "CUDA_VISIBLE_DEVICES=X)",
    )
    parser.add_argument(
        "--num_threads_preprocessing",
        required=False,
        default=6,
        type=int,
        help="Determines many background processes will be used for data preprocessing. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 6",
    )
    parser.add_argument(
        "--num_threads_nifti_save",
        required=False,
        default=2,
        type=int,
        help="Determines many background processes will be used for segmentation export. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 2",
    )
    parser.add_argument(
        "--tta",
        required=False,
        type=int,
        default=1,
        help="Set to 0 to disable test time data "
        "augmentation (speedup of factor "
        "4(2D)/8(3D)), "
        "lower quality segmentations",
    )
    parser.add_argument(
        "--use_gaussian",
        required=False,
        type=int,
        default=1,
        help="Set to 0 to disable gaussian",
    )
    parser.add_argument(
        "--custom_resampling",
        required=False,
        type=int,
        default=0,
        help="Set to 1 to enable our own resampling",
    )
    parser.add_argument(
        "--overwrite_existing",
        required=False,
        type=int,
        default=1,
        help="Set this to 0 if you need "
        "to resume a previous "
        "prediction. Default: 1 "
        "(=existing segmentations "
        "in output_folder will be "
        "overwritten)",
    )
    parser.add_argument("--mode", type=str, default="normal", required=False)
    parser.add_argument(
        "--all_in_gpu",
        type=str,
        default="None",
        required=False,
        help="can be None, False or True",
    )
    parser.add_argument(
        "--step_size", type=float, default=0.5, required=False, help="don't touch"
    )
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations, has no effect if mode=fastest")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z is z is done differently")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False, has no effect if mode=fastest")
    parser.add_argument(
        "--disable_mixed_precision",
        default=False,
        action="store_true",
        required=False,
        help="Predictions are done with mixed precision by default. This improves speed and reduces "
        "the required vram. If you want to disable mixed precision you can set this flag. Note "
        "that this is not recommended (mixed precision is ~2x faster!)",
    )
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.lowres_segmentations == "None":
        args.lowres_segmentations = None

    if isinstance(args.folds, list):
        if args.folds[0] == "all" and len(args.folds) == 1:
            pass
        else:
            args.folds = [int(i) for i in args.folds]
    elif args.folds == "None":
        args.folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    if args.tta == 0:
        args.tta = False
    elif args.tta == 1:
        args.tta = True
    else:
        raise ValueError("Unexpected value for tta, Use 1 or 0")

    if args.use_gaussian == 0:
        args.use_gaussian = False
    elif args.use_gaussian == 1:
        args.use_gaussian = True
    else:
        raise ValueError("Unexpected value for use_gaussian, Use 1 or 0")

    if args.custom_resampling == 0:
        args.custom_resampling = False
    elif args.custom_resampling == 1:
        args.custom_resampling = True
    else:
        raise ValueError("Unexpected value for custom_resampling, Use 1 or 0")

    if args.overwrite_existing == 0:
        args.overwrite_existing = False
    elif args.overwrite_existing == 1:
        args.overwrite_existing = True
    else:
        raise ValueError("Unexpected value for overwrite, Use 1 or 0")

    assert args.all_in_gpu in ["None", "False", "True"]
    if args.all_in_gpu == "None":
        args.all_in_gpu = None
    elif args.all_in_gpu == "True":
        args.all_in_gpu = True
    elif args.all_in_gpu == "False":
        args.all_in_gpu = False

    return args


def predict_from_folder(
    model: str,
    input_folder: str,
    output_folder: str,
    folds: Union[Tuple[int], List[int]],
    save_npz: bool,
    num_threads_preprocessing: int,
    num_threads_nifti_save: int,
    lowres_segmentations: Union[str, None],
    part_id: int,
    num_parts: int,
    tta: bool,
    mixed_precision: bool = True,
    overwrite_existing: bool = True,
    mode: str = "normal",
    overwrite_all_in_gpu: bool = None,
    step_size: float = 0.5,
    checkpoint_name: str = "model_final_checkpoint",
    segmentation_export_kwargs: dict = None,
    disable_postprocessing: bool = False,
    use_gaussian: bool = True,
):
    """This we don't have to optimize, takes only 0.05 seconds"""
    maybe_mkdir_p(output_folder)
    shutil.copyfile(join(model, "plans.pkl"), join(output_folder, "plans.pkl"))

    assert isfile(
        join(model, "plans.pkl")
    ), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))["num_modalities"]

    # check input folder integrity
    case_ids = check_input_folder_and_return_caseIDs(
        input_folder, expected_num_modalities
    )

    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [
        [
            join(input_folder, i)
            for i in all_files
            if i[: len(j)].startswith(j) and len(i) == (len(j) + 12)
        ]
        for j in case_ids
    ]

    if lowres_segmentations is not None:
        assert isdir(
            lowres_segmentations
        ), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [
            join(lowres_segmentations, i + ".nii.gz") for i in case_ids
        ]
        assert all([isfile(i) for i in lowres_segmentations]), (
            "not all lowres_segmentations files are present. "
            "(I was searching for case_id.nii.gz in that folder)"
        )
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    return predict_cases(
        model,
        list_of_lists[part_id::num_parts],
        output_files[part_id::num_parts],
        folds,
        num_threads_preprocessing,
        num_threads_nifti_save,
        lowres_segmentations,
        tta,
        mixed_precision=mixed_precision,
        overwrite_existing=overwrite_existing,
        all_in_gpu=overwrite_all_in_gpu,
        step_size=step_size,
        checkpoint_name=checkpoint_name,
        disable_postprocessing=disable_postprocessing,
        use_gaussian=use_gaussian,
    )


def predict_cases(
    model,
    list_of_lists,
    output_filenames,
    folds,
    num_threads_preprocessing,
    num_threads_nifti_save,
    segs_from_prev_stage=None,
    do_tta=True,
    mixed_precision=True,
    overwrite_existing=False,
    all_in_gpu=False,
    step_size=0.5,
    checkpoint_name="model_final_checkpoint",
    disable_postprocessing: bool = False,
    use_gaussian: bool = True,
):
    # Start off with emptying cuda cache to ensure 0 GPU RAM
    print("emptying cuda cache")
    torch.cuda.empty_cache()

    # Checks on the case filenames etc.
    assert len(list_of_lists) == len(output_filenames)
    if segs_from_prev_stage is not None:
        assert len(segs_from_prev_stage) == len(output_filenames)

    cleaned_output_files = []
    for o in output_filenames:
        dr, f = os.path.split(o)
        if len(dr) > 0:
            maybe_mkdir_p(dr)
        if not f.endswith(".nii.gz"):
            f, _ = os.path.splitext(f)
            f = f + ".nii.gz"
        cleaned_output_files.append(join(dr, f))

    if not overwrite_existing:
        print("number of cases:", len(list_of_lists))
        not_done_idx = [i for i, j in enumerate(cleaned_output_files) if not isfile(j)]

        cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
        list_of_lists = [list_of_lists[i] for i in not_done_idx]
        if segs_from_prev_stage is not None:
            segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print(
            "number of cases that still need to be predicted:",
            len(cleaned_output_files),
        )

    # This step takes noticable time, about 4 seconds per fold
    # This takes about 1.4 GB of gpu space
    # And also loads model to RAM?
    # I think loading params before preprocessing is a waste of time
    trainer, params = load_model_and_checkpoint_files(
        model,
        folds,
        mixed_precision=mixed_precision,
        checkpoint_name=checkpoint_name,
        load_params=False,
        init_trainer=False,
    )

    # This step takes noticable time, about 10 seconds
    # We can not make this quicker with our own preprocessing
    print("running preprocessing")
    preprocessing = preprocess(
        trainer,
        list_of_lists,
        cleaned_output_files,
        segs_from_prev_stage,
    )

    print("starting prediction...")
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed

        # Next part is GPU inference
        print("predicting", output_filename)
        p = params[0]
        trainer.load_checkpoint(p, False)
        res = trainer.predict_preprocessed_data_return_seg_and_softmax(
            d,
            do_mirroring=do_tta,
            mirror_axes=trainer.data_aug_params["mirror_axes"],
            use_sliding_window=True,
            step_size=step_size,
            use_gaussian=use_gaussian,
            all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision,
        )
        print("emtpyting torch cache")
        torch.cuda.empty_cache()

        print("aggregating predictions")
        seg = res[0]

        print("applying transpose_backward")
        transpose_forward = trainer.plans.get("transpose_forward")
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get("transpose_backward")
            seg = seg.transpose([i for i in transpose_backward])

        print("exporting segmentation")
        save_segmentation_nifti(seg, output_filename, dct, 0, None)
        print("done")


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    parser = setup_argparse()
    args = parse_args(parser)
    tic = time.time()

    nnunet_input_folder = args.input_folder

    # nnunet predict
    predict_from_folder(
        args.model_output_folder,
        nnunet_input_folder,
        args.output_folder,
        args.folds,
        args.save_npz,
        args.num_threads_preprocessing,
        args.num_threads_nifti_save,
        args.lowres_segmentations,
        args.part_id,
        args.num_parts,
        args.tta,
        mixed_precision=not args.disable_mixed_precision,
        overwrite_existing=args.overwrite_existing,
        mode=args.mode,
        overwrite_all_in_gpu=args.all_in_gpu,
        step_size=args.step_size,
        use_gaussian=args.use_gaussian,
    )

    toc = time.time()
    print()
    print(f"Prediction took {toc-tic}")
