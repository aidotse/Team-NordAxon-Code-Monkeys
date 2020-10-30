import glob
from typing import List

from joblib import Parallel, delayed
# from natsort import natsorted, ns
import numpy as np
from skimage import io
from skimage import morphology


class MaskSlic:
    """
    Apply MaskSlic
    :param path: Any valid string path or os.PathLike object.
    """
    def __init__(self, path=None, output_path=None):
        self.path = path
        self.output_path = output_path

    def apply_maskslic(self):
        """
        Apply MaskSlic to file paths in list returned by get_files_paths()
        """
        images_paths = self.get_files_paths()

        for img_path in images_paths:
            img_obj = io.imread(img_path)
            mask = self.mask_slic(img_obj)
            io.imsave('{}/{}_mask.tif'.format(self.output_path, img_path.name), mask)

    def apply_maskslic_parallel(self):
        """
        Apply MaskSlic to images in paralell
        """
        images_np_lst = self.read_images()
        Parallel(n_jobs=4)(delayed(self.mask_slic)(i) for i in images_np_lst)

    # def apply_maskslic_V2(self):
    #     """
    #     Apply MaskSlic to files found in raw data folder
    #     """
    #     filename = ...
    #     images_np_lst = self.read_images()
    #     for image_obj in images_np_lst:
    #         mask = self.mask_slic(image_obj)
    #         io.imsave('{}/{}.tif'.format(self.output_path, filename), mask)

    def get_files_paths(self) -> List[str]:
        """
        Create list with files/images paths found in raw data directory/input path

        Returns:
        list: list with paths to each raw image found in raw data dir
        """
        files_paths = [f for f in glob.glob(self.path + "/*.tif", recursive=False)]

        return files_paths

    def read_images(self) -> List[np.ndarray]:
        """
        Create list with images/numpy objects from paths returned by get_file_paths()

        Returns:
        list: list with paths to each raw image found in raw data dir
        """
        raw_imgs_paths = self.get_file_paths()
        img_objs_list = [io.imread(file_path) for file_path in raw_imgs_paths]

        return img_objs_list

    def mask_slic(self, img) -> np.ndarray:
        """
        Scikit-image's MaskSlic

        Returns:
        mask: scikit-image object (ndarray)
        """
        mask = morphology.remove_small_holes(
            morphology.remove_small_objects(
                img > 2*np.mean(img), 500), 500)

        mask = morphology.opening(mask, morphology.disk(3))

        return mask
