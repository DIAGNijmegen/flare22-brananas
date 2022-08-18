import SimpleITK as sitk
import numpy as np
import json
import os
import time
import random
import wandb
import psutil as p
from pynvml import nvmlDeviceGetCount
from pynvml.smi import nvidia_smi
import signal


def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False, out_size=[]):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    if not out_size:
        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))),
        ]

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    # perform resampling
    itk_image = resample.Execute(itk_image)

    return itk_image


def FixLabel(resampled_label):
    resampled_label_np = sitk.GetArrayFromImage(resampled_label)
    resampled_label_np_fixed = np.copy(resampled_label_np)
    for i in range(resampled_label_np.shape[0]):
        if np.all(resampled_label_np[i, :, :]):
            resampled_label_np_fixed[i, :, :] = 0
            # resampled_label_np[i, :, :] = 0

    for i in range(resampled_label_np.shape[1]):
        if np.all(resampled_label_np[:, i, :]):
            resampled_label_np_fixed[:, i, :] = 0
            # resampled_label_np[:, i, :] = 0

    for i in range(resampled_label_np.shape[2]):
        if np.all(resampled_label_np[:, :, i]):
            resampled_label_np_fixed[:, :, i] = 0
            # resampled_label_np[:, :, i] = 0
    resampled_label_fixed = sitk.GetImageFromArray(resampled_label_np_fixed)
    # resampled_label_fixed = sitk.GetImageFromArray(resampled_label_np)
    return resampled_label_fixed


def Generate_nnUnet_dataset_json(dataset_path):
    nnunett_masks_path = os.path.join(dataset_path, "labelsTr")
    nnunett_images_path = os.path.join(dataset_path, "imagesTr")

    dataset = {}
    dataset["name"] = dataset_path.split("/")[-1]
    dataset["description"] = "FLARE22"
    dataset["reference"] = "Radboud University Medical Center "
    dataset["licence"] = "XXXX"
    dataset["relase"] = "XXXX"
    dataset["tensorImageSize"] = "3D"
    dataset["modality"] = {"0": "CT"}
    dataset["labels"] = {
        "0": "background",
        "1": "liver",
        "2": "right kidney",
        "3": "spleen",
        "4": "pancreas",
        "5": "aorta",
        "6": "inferior vena cava",
        "7": "right adrenal gland",
        "8": "left adrenal gland",
        "9": "gallbladder",
        "10": "esophagus",
        "11": "stomach",
        "12": "duodenum",
        "13": "left kidney",
    }

    dataset["numTraining"] = len(os.listdir(nnunett_masks_path))
    dataset["numTest"] = 0
    dataset["training"] = []

    c = 0
    for f in os.scandir(nnunett_images_path):
        c = c + 1
        json_image = "./" + "imagesTr/" + f.name[:-12] + ".nii.gz"
        json_mask = "./" + "labelsTr/" + f.name[:-12] + ".nii.gz"

        dataset["training"].append({"image": json_image, "label": json_mask})

    dataset["test"] = []

    json_path = os.path.join(dataset_path, "dataset.json")
    with open(json_path, "w") as outfile:
        json.dump(dataset, outfile)


def ensemble_predictions(predictions):
    """Generate an ensembled mask from raw softmax predictions

    Parameters
    ----------
    predictions: list
        List of numpy arrays
    """
    # Stack volumes, shape becomes (5, 14, z, y, x)
    prediction = np.stack(predictions)

    # Average over all 5 models
    prediction = np.mean(prediction, axis=0)

    # Argmax
    prediction = np.argmax(prediction, axis=0)

    return prediction


def load_nifti(path):
    image_sitk = sitk.ReadImage(str(path))
    image = sitk.GetArrayFromImage(image_sitk)
    return image


def load_softmax(softmax_path):
    softmax = np.load(softmax_path)
    softmax = softmax["softmax"]
    return softmax


def get_cpu_usage():
    """Returns CPU usage in %"""
    return p.cpu_percent(interval=0.1)


def get_RAM():
    """Returns RAM usage in GB"""
    return p.virtual_memory().used / 1073741824


def get_gpu_usage():
    """Return GPU usage in GB"""
    gpu_index = 0
    nvsmi = nvidia_smi.getInstance()
    dictm = nvsmi.DeviceQuery("memory.free, memory.total")
    gpu_memory = (
        dictm["gpu"][gpu_index]["fb_memory_usage"]["total"]
        - dictm["gpu"][gpu_index]["fb_memory_usage"]["free"]
    )
    # print(gpu_memory)
    return gpu_memory / 1024


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


def log_resources(queue):
    os.environ["WANDB_API_KEY"] = ""
    run = wandb.init(project="flare22")
    killer = GracefulKiller()

    # Signal that logger is setup
    queue.put(True)

    # Start logging
    while not killer.kill_now:
        time.sleep(0.1)
        run.log(
            {
                "CPU usage (%)": get_cpu_usage(),
                "RAM (GB)": get_RAM(),
                "GPU usage (GB)": get_gpu_usage(),
            }
        )

    wandb.finish()
