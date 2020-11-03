import os
import shutil
from pathlib import Path

import cv2

from utils.utils import get_image_metadata


def divide_astra_zeneca_train_validation(
    input_path:str = "01_raw",
    output_path:str = "02_train_valid_split", 
    train_ratio=0.8
    ) ->  None:
    """Divide images for the Astra Zeneca competition into training and validation sets.

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

    dataset_samples = glob.glob(os.path.join(input_path, "*/*/Assay*"))

    dataset_dicts = [get_image_metadata(path) for path in dataset_samples]

    # Group all 7 inputs with all 3 respective targets into variable sample
    samples = dict()
    for sample_dict in dataset_dicts:
        sample_key = (sample_dict["row_col"], sample_dict["field of view"])

        if samples.get(sample_key) is None:
            samples[sample_key] = {
                "input": dict(),
                "target": dict()
            }

        if sample_dict["action_list_number"] == "A04": # or sample_dict["imaging_channel"] == "C04"
            # Is an input
            z_number_3d = sample_dict["z_number_3d"]
            samples[sample_key]["input"][z_number_3d] = sample_dict["path"]
        else:
            # Is an target
            action_list_number = sample_dict["action_list_number"]
            samples[sample_key]["target"][action_list_number] = sample_dict["path"]

    samples = list(samples.values())

    # Modulo
    if os.path.isdir(os.path.join(output_path, "train")):
        shutil.rmtree(os.path.join(output_path, "train"))
    if os.path.isdir(os.path.join(output_path, "valid")):
        shutil.rmtree(os.path.join(output_path, "valid"))

    Path(os.path.join(output_path, "valid", "input")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path, "valid", "targets")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path, "train", "input")).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path, "train", "targets")).mkdir(exist_ok=True, parents=True)

    crop_size = (256, 256)

    for idx in range(len(samples)):
        sample_dict = samples[idx]

        w, h = cv2.imread(sample_dict["input"]["Z01"], -1).shape
        assert crop_size[0] <= w
        assert crop_size[1] <= h

        input = torch.zeros((7, w, h))
        output = torch.zeros((3, w, h))
        for i, z_number_3d in enumerate(["Z01", "Z02", "Z03", "Z04", "Z05", "Z06", "Z07"]):
            img_path = sample_dict["input"][z_number_3d]

            img = cv2.imread(img_path, -1)

            w, h = img.shape
            
            cut_off_idx = int(w*train_ratio)

            train_img = img[:cut_off_idx]
            valid_img = img[cut_off_idx:]

            filename = os.path.basename(img_path)
            save_path_train = os.path.join(output_path, "train", "input", filename)
            save_path_valid = os.path.join(output_path, "valid", "input", filename)
            cv2.imwrite(save_path_train, train_img)
            cv2.imwrite(save_path_valid, valid_img)

        for i, action_list_number in enumerate(["A01", "A02", "A03"]):
            img_path = sample_dict["target"][action_list_number]

            img = cv2.imread(img_path, -1)

            cut_off_idx = int(w*train_ratio)

            train_img = img[:cut_off_idx]
            valid_img = img[cut_off_idx:]

            filename = os.path.basename(img_path)
            save_path_train = os.path.join(output_path, "train", "targets", filename)
            save_path_valid = os.path.join(output_path, "valid", "targets", filename)
            cv2.imwrite(save_path_train, train_img)
            cv2.imwrite(save_path_valid, valid_img)

if __name__ == "__main__":
    divide_astra_zeneca_train_validation()