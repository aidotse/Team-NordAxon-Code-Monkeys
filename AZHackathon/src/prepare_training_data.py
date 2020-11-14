import os
import shutil
import glob

import pandas as pd
import numpy as np
from skimage import morphology
from tqdm import tqdm
from pathlib import Path
import cv2

from utils.utils import get_image_metadata
"""    
Group by row_col and field of view
    # row_col
    # field of view
    Input and Target share these common values:
    - row_col       = sample id? 
    - field of view = amount of zoom
    For identifying INPUT:
    - action_list_number A04
    - imaging_channel    C04
    - z_number_3d        Z01 - Z07
    For identifying TARGET:
    - action_list_number A01 A02 and A03
    - imaging_channel    C01, C02, C03
    - z_number_3d        Z01
"""
def divide_input_target(
        input_path:str = "../data/01_raw/",
        output_path:str = "../data/02_intermediate/"
        ) -> None:
    """Divide images for the Astra Zeneca competition into training and validation sets.
    """

    dataset_samples = glob.glob(os.path.join(input_path, "*/Assay*"))
    print(f"Dataset contains {len(dataset_samples)} .tif files")
    dataset_dicts = [get_image_metadata(path) for path in dataset_samples]

    # Group all 7 inputs with all 3 respective targets into variable sample
    samples = dict()
    for sample_dict in dataset_dicts:
        magnification = os.path.basename(os.path.dirname(sample_dict["path"]))
        sample_key = (sample_dict["row_col"], sample_dict["field of view"], magnification)
        if samples.get(sample_key) is None:
            samples[sample_key] = {"input": dict(), "target": dict()}
        if sample_dict["action_list_number"] == "A04": # or sample_dict["imaging_channel"] == "C04"
            # Is an input
            z_number_3d = sample_dict["z_number_3d"]
            samples[sample_key]["input"][z_number_3d] = sample_dict["path"]
        else:
            # Is a target
            action_list_number = sample_dict["action_list_number"]
            samples[sample_key]["target"][action_list_number] = sample_dict["path"]
    samples = list(samples.values())

    print(f"Dataset contains {len(samples)} samples (1 sample = 7 brightfield and 3 fluorescent)")

    shutil.rmtree(os.path.join(output_path, "input"))
    shutil.rmtree(os.path.join(output_path, "targets"))
    Path(os.path.join(output_path, "input")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path, "targets")).mkdir(exist_ok=True, parents=True)
    for idx in tqdm(range(len(samples))):
        sample_dict = samples[idx]
        w, h = cv2.imread(sample_dict["input"]["Z01"], -1).shape

        magnification = os.path.basename(os.path.dirname(sample_dict["input"]["Z01"]))
        Path(os.path.join(output_path, "input", magnification)).mkdir(exist_ok=True, parents=True)
        Path(os.path.join(output_path, "targets", magnification)).mkdir(exist_ok=True, parents=True)

        for i, z_number_3d in enumerate(["Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07"]):
            img_path = sample_dict["input"][z_number_3d]
            img = cv2.imread(img_path, -1)
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_path, "input", magnification, filename)
            cv2.imwrite(save_path, img)

        for i, action_list_number in enumerate(["A01", "A02", "A03"]):
            img_path = sample_dict["target"][action_list_number]
            img = cv2.imread(img_path, -1)
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_path, "targets", magnification, filename)
            cv2.imwrite(save_path, img)



def mask(img:np.ndarray) -> np.ndarray:
    """
    Scikit-image's Mask

    Returns:
    mask: scikit-image object (ndarray)
    """
    mask = (img > 2*img.mean()).astype(np.float)
    
    mask = morphology.remove_small_holes(
        morphology.remove_small_objects(
            img > 2*np.mean(img), 500), 500)

    mask = morphology.opening(mask, morphology.disk(3))

    return mask

def create_masks(input_path:str = "../data/02_intermediate/", 
                 output_path:str = "../data/02_intermediate/") -> None:
    Path(os.path.join(input_path, "masks", "20x_images")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(input_path, "masks", "40x_images")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(input_path, "masks", "60x_images")).mkdir(exist_ok=True, parents=True)
    target_images = [path for path in glob.glob(os.path.join(input_path, "targets", "*/*")) if "A01" in path]
    for target_path in tqdm(target_images):
        img = cv2.imread(target_path, -1)
        img_mask = mask(img)
        save_path = target_path.replace("/targets/", "/masks/")
        img_mask = img_mask.astype(np.int8)
        success = cv2.imwrite(save_path, img_mask)
        if not success:
            print(f"Could not save {save_path}")


def split_train_validation(
        input_path:str = "../data/02_intermediate/",
        output_path:str = "../data/03_training_data/",
        normalize:str = None,
        stats_csv_path:str = None,
    ) -> None:
    
    if (normalize is not None) or (stats_csv_path is not None):
        assert normalize is not None # Please define the path to the csv file
        assert stats_csv_path is not None # Please define normalization method. Options=[by_magnification, global]

    train_wells = ["D02", "D03", "D04", "C02", "C03", "C04"]
    valid_wells = ["B03", "B04"]

    dataset_samples = glob.glob(os.path.join(input_path, "*/*/Assay*"))
    print(f"Dataset contains {len(dataset_samples)} .tif files")
    dataset_dicts = [get_image_metadata(path) for path in dataset_samples]

    # Group all 7 inputs with all 3 respective targets into variable sample
    samples = dict()
    unique_wells = list()
    for sample_dict in dataset_dicts:
        unique_wells.append(sample_dict["row_col"])
        magnification = os.path.basename(os.path.dirname(sample_dict["path"]))
        sample_key = (sample_dict["row_col"], sample_dict["field of view"], magnification)

        if samples.get(sample_key) is None:
            samples[sample_key] = {"input": dict(), "target": dict(), "mask": dict(), "well": None}

        samples[sample_key]["well"] = sample_dict["row_col"]

        if sample_dict["action_list_number"] == "A04" and "input" in sample_dict["path"]: 
            # Is an input
            z_number_3d = sample_dict["z_number_3d"]
            samples[sample_key]["input"][z_number_3d] = sample_dict["path"]
        elif "targets" in sample_dict["path"]:
            # Is a target
            action_list_number = sample_dict["action_list_number"]
            samples[sample_key]["target"][action_list_number] = sample_dict["path"]
        elif "masks" in sample_dict["path"]:
            # Is a mask
            action_list_number = sample_dict["action_list_number"]
            samples[sample_key]["mask"][action_list_number] = sample_dict["path"]
        else:
            print("This is not supposed to be reached")
            raise Error()
    samples = list(samples.values())
    unique_row_col = set(unique_wells)

    if normalize is not None:
        stats = pd.read_csv(stats_csv_path)
        stats = stats.rename(columns={"Unnamed: 0": "row"}).set_index("row")

    print(f"Dataset contains {len(samples)} samples (1 sample = 7 brightfield and 3 fluorescent)")
    print("All wells:", set(unique_row_col))

    Path(os.path.join(output_path, "train/input")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path, "train/targets")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path, "valid/input")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path, "valid/targets")).mkdir(exist_ok=True, parents=True)

    for idx in tqdm(range(len(samples))):
        sample_dict = samples[idx]

        if sample_dict["well"] in train_wells:
            _set = "train"
        elif sample_dict["well"] in valid_wells:
            _set = "valid"
        else:
            assert False # This is not supposed to be reached

        magnification = os.path.basename(os.path.dirname(sample_dict["input"]["Z01"]))
        Path(os.path.join(output_path, _set, "input", magnification)).mkdir(exist_ok=True, parents=True)
        Path(os.path.join(output_path, _set, "targets", magnification)).mkdir(exist_ok=True, parents=True)
        Path(os.path.join(output_path, _set, "masks", magnification)).mkdir(exist_ok=True, parents=True)

        for i, z_number_3d in enumerate(["Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07"]):
            img_path = sample_dict["input"][z_number_3d]
            img = cv2.imread(img_path, -1)
            
            if normalize == "by_magnification":
                img=(img - stats.loc[magnification]["mean"])/stats.loc[magnification]["std"]
            elif normalize == "global":
                img=(img - stats.loc["global"]["mean"])/stats.loc["global"]["std"]
                
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_path, _set, "input", magnification, filename)
            cv2.imwrite(save_path, img)

        for i, action_list_number in enumerate(["A01", "A02", "A03"]):
            img_path = sample_dict["target"][action_list_number]
            img = cv2.imread(img_path, -1)
            filename = os.path.basename(img_path)
            save_path = os.path.join(output_path, _set, "targets", magnification, filename)
            cv2.imwrite(save_path, img)

        img_path = sample_dict["mask"]["A01"]
        img = cv2.imread(img_path, -1)
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_path, _set, "masks", magnification, filename)
        cv2.imwrite(save_path, img)


if __name__ == "__main__":
    input_path = "../data/01_raw/"
    output_path = "../data/02_intermediate/"
    divide_input_target(input_path, output_path)
    
    input_path = "../data/02_intermediate/"
    output_path = "../data/02_intermediate/"
    create_masks(input_path, output_path)
    
    input_path = "../data/02_intermediate/"
    output_path = "../data/03_training_data/"
    split_train_validation(input_path, output_path)
    
    input_path = "../data/02_intermediate/"
    output_path = "../data/03_training_data/"
    normalize = "by_magnification"
    stats_csv_path = "../data/06_outputs/input_statistics.csv"
    split_train_validation(input_path, output_path, normalize, stats_csv_path)
