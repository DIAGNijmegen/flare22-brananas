#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import nnunet
import torch
from batchgenerators.utilities.file_and_folder_operations import *
import importlib
import pkgutil
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from collections import OrderedDict
from nnunet.network_architecture.generic_UNet import Generic_UNet
from torch import nn
import numpy as np
from typing import Tuple, List
import time
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper

from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List

from torch.cuda.amp import autocast


class Generic_UNet(Generic_UNet):
    def _internal_predict_3D_3Dconv_tiled(
        self,
        x: np.ndarray,
        step_size: float,
        do_mirroring: bool,
        mirror_axes: tuple,
        patch_size: tuple,
        regions_class_order: tuple,
        use_gaussian: bool,
        pad_border_mode: str,
        pad_kwargs: dict,
        all_in_gpu: bool,
        verbose: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        print("USING CUSTOM INFERENCE THING!")
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose:
            print("step_size:", step_size)
        if verbose:
            print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(
            x, patch_size, pad_border_mode, pad_kwargs, True, None
        )
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(
            patch_size, data_shape[1:], step_size
        )
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]
            ):
                if verbose:
                    print("computing Gaussian")
                gaussian_importance_map = self._get_gaussian(
                    patch_size, sigma_scale=1.0 / 8
                )

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose:
                    print("done")
            else:
                if verbose:
                    print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            # predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(
                    self.get_device(), non_blocking=True
                )

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[
                    gaussian_importance_map == 0
                ] = gaussian_importance_map[gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose:
                print("initializing result array (on GPU)")
            aggregated_results = torch.zeros(
                [self.num_classes] + list(data.shape[1:]),
                dtype=torch.half,
                device=self.get_device(),
            )

            if verbose:
                print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose:
                print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros(
                [self.num_classes] + list(data.shape[1:]),
                dtype=torch.half,
                device=self.get_device(),
            )

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros(
                [self.num_classes] + list(data.shape[1:]), dtype=np.float32
            )
            aggregated_nb_of_predictions = np.zeros(
                [self.num_classes] + list(data.shape[1:]), dtype=np.float32
            )

        cleared_cache = False
        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z],
                        mirror_axes,
                        do_mirroring,
                        gaussian_importance_map,
                    )[0]

                    # Added this to clear cuda cache as soon as possible
                    if not cleared_cache:
                        torch.cuda.empty_cache()
                        cleared_cache = True

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[
                        :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z
                    ] += predicted_patch
                    aggregated_nb_of_predictions[
                        :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z
                    ] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [
                slice(0, aggregated_results.shape[i])
                for i in range(len(aggregated_results.shape) - (len(slicer) - 1))
            ]
            + slicer[1:]
        )
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = aggregated_results.detach().cpu().numpy()
            else:
                class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(
                class_probabilities_here.shape[1:], dtype=np.float32
            )
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose:
                print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            aggregated_results = aggregated_results.detach().cpu().numpy()

        if verbose:
            print("prediction done")
        return predicted_segmentation, aggregated_results


class nnUNetTrainer(nnUNetTrainer):
    def load_checkpoint_ram(self, checkpoint, train=True):
        print("USING CUSTOM TRAINER")
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint["state_dict"].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith("module."):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if train:
                if "amp_grad_scaler" in checkpoint.keys():
                    self.amp_grad_scaler.load_state_dict(checkpoint["amp_grad_scaler"])

        self.network.load_state_dict(new_state_dict)
        # self.epoch = checkpoint["epoch"]
        self.epoch = 1000
        if train:
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if (
                self.lr_scheduler is not None
                and hasattr(self.lr_scheduler, "load_state_dict")
                and checkpoint["lr_scheduler_state_dict"] is not None
            ):
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        # (
        #     self.all_tr_losses,
        #     self.all_val_losses,
        #     self.all_val_losses_tr_mode,
        #     self.all_val_eval_metrics,
        # ) = checkpoint["plot_stuff"]

        # load best loss (if present)
        if "best_stuff" in checkpoint.keys():
            (
                self.best_epoch_based_on_MA_tr_loss,
                self.best_MA_tr_loss_for_patience,
                self.best_val_eval_criterion_MA,
            ) = checkpoint["best_stuff"]

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        # if self.epoch != len(self.all_tr_losses):
        #     self.print_to_log_file(
        #         "WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
        #         "due to an old bug and should only appear when you are loading old models. New "
        #         "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)"
        #     )
        #     self.epoch = len(self.all_tr_losses)
        #     self.all_tr_losses = self.all_tr_losses[: self.epoch]
        #     self.all_val_losses = self.all_val_losses[: self.epoch]
        #     self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[: self.epoch]
        #     self.all_val_eval_metrics = self.all_val_eval_metrics[: self.epoch]

        self._maybe_init_amp()

        # TODO I added this myself, it seems to make GPU usage less...
        self.network.half()

    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """
        # self.print_to_log_file(self.net_num_pool_op_kernel_sizes)
        # self.print_to_log_file(self.net_conv_kernel_sizes)

        net_numpool = len(self.net_num_pool_op_kernel_sizes)

        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {"eps": 1e-5, "affine": True}
        dropout_op_kwargs = {"p": 0, "inplace": True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        self.network = Generic_UNet(
            self.num_input_channels,
            self.base_num_features,
            self.num_classes,
            net_numpool,
            self.conv_per_stage,
            2,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            net_nonlin,
            net_nonlin_kwargs,
            False,
            False,
            lambda x: x,
            InitWeights_He(1e-2),
            self.net_num_pool_op_kernel_sizes,
            self.net_conv_kernel_sizes,
            False,
            True,
            True,
        )
        self.network.inference_apply_nonlin = softmax_helper

        if torch.cuda.is_available():
            self.network.cuda()

    def predict_preprocessed_data_return_seg_and_softmax(
        self,
        data: np.ndarray,
        do_mirroring: bool = True,
        mirror_axes: Tuple[int] = None,
        use_sliding_window: bool = True,
        step_size: float = 0.5,
        use_gaussian: bool = True,
        pad_border_mode: str = "constant",
        pad_kwargs: dict = None,
        all_in_gpu: bool = False,
        verbose: bool = True,
        mixed_precision: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == "constant" and pad_kwargs is None:
            pad_kwargs = {"constant_values": 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params["mirror_axes"]

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], (
                "Cannot do mirroring as test time augmentation when training "
                "was done without mirroring"
            )

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(
            data,
            do_mirroring=do_mirroring,
            mirror_axes=mirror_axes,
            use_sliding_window=use_sliding_window,
            step_size=step_size,
            patch_size=self.patch_size,
            regions_class_order=self.regions_class_order,
            use_gaussian=use_gaussian,
            pad_border_mode=pad_border_mode,
            pad_kwargs=pad_kwargs,
            all_in_gpu=all_in_gpu,
            verbose=verbose,
            mixed_precision=mixed_precision,
        )
        self.network.train(current_mode)
        return ret


def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(
                    [join(folder[0], modname)],
                    trainer_name,
                    current_module=next_current_module,
                )
            if tr is not None:
                break

    return tr


def restore_model(pkl_file, checkpoint=None, train=False, fp16=None):
    """
    This is a utility function to load any nnUNet trainer from a pkl. It will recursively search
    nnunet.trainig.network_training for the file that contains the trainer and instantiate it with the arguments saved in the pkl file. If checkpoint
    is specified, it will furthermore load the checkpoint file in train/test mode (as specified by train).
    The pkl file required here is the one that will be saved automatically when calling nnUNetTrainer.save_checkpoint.
    :param pkl_file:
    :param checkpoint:
    :param train:
    :param fp16: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    info = load_pickle(pkl_file)
    init = info["init"]
    name = info["name"]
    search_in = join(nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class(
        [search_in], name, current_module="nnunet.training.network_training"
    )

    if tr is None:
        """
        Fabian only. This will trigger searching for trainer classes in other repositories as well
        """
        try:
            import meddec

            search_in = join(meddec.__path__[0], "model_training")
            tr = recursive_find_python_class(
                [search_in], name, current_module="meddec.model_training"
            )
        except ImportError:
            pass

    if tr is None:
        raise RuntimeError(
            "Could not find the model trainer specified in checkpoint in nnunet.trainig.network_training. If it "
            "is not located there, please move it or change the code of restore_model. Your model "
            "trainer can be located in any directory within nnunet.trainig.network_training (search is recursive)."
            "\nDebug info: \ncheckpoint file: %s\nName of trainer: %s "
            % (checkpoint, name)
        )
    # assert issubclass(tr, nnUNetTrainer), (
    #     "The network trainer was found but is not a subclass of nnUNetTrainer. "
    #     "Please make it so!"
    # )

    # this is now deprecated
    """if len(init) == 7:
        print("warning: this model seems to have been saved with a previous version of nnUNet. Attempting to load it "
              "anyways. Expect the unexpected.")
        print("manually editing init args...")
        init = [init[i] for i in range(len(init)) if i != 2]"""

    # ToDo Fabian make saves use kwargs, please...

    tr = nnUNetTrainer
    trainer = tr(*init)

    # We can hack fp16 overwriting into the trainer without changing the init arguments because nothing happens with
    # fp16 in the init, it just saves it to a member variable
    if fp16 is not None:
        trainer.fp16 = fp16

    trainer.process_plans(info["plans"])
    # if checkpoint is not None:
    #     trainer.load_checkpoint(checkpoint, train)
    return trainer


def load_best_model_for_inference(folder):
    checkpoint = join(folder, "model_best.model")
    pkl_file = checkpoint + ".pkl"
    return restore_model(pkl_file, checkpoint, False)


def load_model_and_checkpoint_files(
    folder,
    folds=None,
    mixed_precision=None,
    checkpoint_name="model_best",
    load_params=True,
    init_trainer=True,
):
    """
    used for if you need to ensemble the five models of a cross-validation. This will restore the model from the
    checkpoint in fold 0, load all parameters of the five folds in ram and return both. This will allow for fast
    switching between parameters (as opposed to loading them from disk each time).

    This is best used for inference and test prediction
    :param folder:
    :param folds:
    :param mixed_precision: if None then we take no action. If True/False we overwrite what the model has in its init
    :return:
    """
    if isinstance(folds, str):
        folds = [join(folder, "all")]
        assert isdir(folds[0]), "no output folder for fold %s found" % folds
    elif isinstance(folds, (list, tuple)):
        if len(folds) == 1 and folds[0] == "all":
            folds = [join(folder, "all")]
        else:
            folds = [join(folder, "fold_%d" % i) for i in folds]
        assert all(
            [isdir(i) for i in folds]
        ), "list of folds specified but not all output folders are present"
    elif isinstance(folds, int):
        folds = [join(folder, "fold_%d" % folds)]
        assert all([isdir(i) for i in folds]), (
            "output folder missing for fold %d" % folds
        )
    elif folds is None:
        print(
            "folds is None so we will automatically look for output folders (not using 'all'!)"
        )
        folds = subfolders(folder, prefix="fold")
        print("found the following folds: ", folds)
    else:
        raise ValueError(
            "Unknown value for folds. Type: %s. Expected: list of int, int, str or None",
            str(type(folds)),
        )

    trainer = restore_model(
        join(folds[0], "%s.model.pkl" % checkpoint_name), fp16=mixed_precision
    )
    trainer.output_folder = folder
    trainer.output_folder_base = folder
    trainer.update_fold(0)
    # Turning this off prevents loading stuff to GPU before inference
    if init_trainer:
        trainer.initialize(False)
    all_best_model_files = [join(i, "%s.model" % checkpoint_name) for i in folds]
    print("using the following model files: ", all_best_model_files)
    if load_params:
        all_params = [
            torch.load(i, map_location=torch.device("cpu"))
            for i in all_best_model_files
        ]
    else:
        all_params = all_best_model_files
    return trainer, all_params


if __name__ == "__main__":
    pkl = "/home/fabian/PhD/results/nnUNetV2/nnUNetV2_3D_fullres/Task004_Hippocampus/fold0/model_best.model.pkl"
    checkpoint = pkl[:-4]
    train = False
    trainer = restore_model(pkl, checkpoint, train)
