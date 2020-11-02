from pathlib import Path
from typing import List, Dict

import numpy as np
from skimage import img_as_ubyte
from skimage import io
from skimage import morphology


class Mask:
    """
    Apply Mask to each magnification's target images
    Expected folders structure in input_path:
    input_path
        - magnification1
            - inputs
            - target
        - magnification2
            - inputs
            - target
        ...
    :param input_path: valid string path or os.PathLike object that contains
                       target images per magnification (details above)
    :param output_path: valid string path or os.PathLike object to store
                        output masks for each magnification found in input_path
    """
    def __init__(self, input_path=None, output_path=None):
        self.input_path = input_path
        self.output_path = output_path
        self.images = None
        self.target_paths = None
        self.target_masks = None

    def store_result(self):
        """
        TODO
        """
        self._apply_mask()
        # add here code to write dict values (final masks) to output_path
        [[io.imsave(fname=img_nm.format(), arr=img_as_ubyte(image_mask)) for img_nm,image_mask in v.items()] for k,v in self.target_masks.items()]

    def _apply_mask(self):
        """
        Apply Mask to target image objects per magnification.

        Outputs dictionary with magnification names as keys (e.g., 20x, 40x).
        Each key's value is a dictionary whose keys are image names (Path().name)
        and values are corresponding image masks (np.ndarray)

        Returns:
        target_masks: nested dict with magnification_x:{'image_name':image_obj_mask} as key:value
        """
        self._read_images()
        target_masks = {magnif_name: {img_nm: self._mask(img_obj) for img_nm, img_obj in img_paths.items()}
                        for magnif_name, img_paths in self.images.items()}

        self.target_masks = target_masks

    def _mask(self, img) -> np.ndarray:
        """
        Scikit-image's Mask

        Returns:
        mask: scikit-image object (ndarray)
        """
        mask = morphology.remove_small_holes(
            morphology.remove_small_objects(
                img > 2*np.mean(img), 500), 500)

        mask = morphology.opening(mask, morphology.disk(3))

        return mask

    def _read_images(self) -> Dict[str, List[np.ndarray]]:
        """
        Create dictionary with magnification names as keys (e.g., 20x, 40x).
        Each key's value is a dictionary whose keys are image names (Path().name)
        and values are corresponding image objects (scikit-image IO)

        Returns:
        magn_img_objs: nested dict with magnification_x:{'image_name':image_obj} as key:value
        """
        self._get_target_files_paths()
        magn_img_objs = {magnif_name: {img.name: io.imread(img) for img in img_paths}
                         for magnif_name, img_paths in self.target_paths.items()}

        self.images = magn_img_objs

    def _get_target_files_paths(self) -> Dict[str, List[str]]:
        """
        Create dictionary with magnification names as keys (e.g., 20x, 40x).
        Each key contains a list with paths to target images/files
        for a magnification.

        Returns:
        magn_target_paths: dict with magnification:[target_paths] as key:value
        """
        target_folders = [i for i in Path(self.input_path).glob("*/*")
                          if "targets" in Path(i).name.lower()]
        magn_target_paths = {}
        for folder_path in target_folders:
            magn_target_paths[Path(folder_path).parent.name] = list(folder_path.glob('*.tif'))

        self.target_paths = magn_target_paths
