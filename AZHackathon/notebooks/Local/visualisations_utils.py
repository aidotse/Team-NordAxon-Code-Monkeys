"""
    Written by NordAxon Code Monkeys 2020-11-08
"""

import os
import numpy as np
import cv2
from collections import defaultdict
from collections import Counter


def get_sample_paths(paths : list, mags : list) -> tuple:
    """
    Structures the input and target paths, given the magnifications and all paths possible
    """
    all_inputs, all_targets = defaultdict(), defaultdict()
    for mag in mags:
        inputs, targets = [], []
        for path in paths:
            if "input" in path and mag in path:
                inputs.append(path)
            if "target" in path and mag in path:
                targets.append(path)
        all_inputs[mag] = inputs
        all_targets[mag] = targets
    return all_inputs, all_targets


def get_specs(mag : str, spec : str) -> list:
    """
    Creates the specific strings for the chosen data attribute "spec"
    """
    if spec == "F":
        if mag == "20x": max = 6
        if mag == "40x": max = 8
        if mag == "60x": max = 12
        specs = ["F"+str(i).zfill(3) for i in range(1,max+1)]
    if spec == "Z":
        specs = ["Z"+str(i).zfill(2) for i in range(1,8)]
    if spec == "A":
        specs = ["A"+str(i).zfill(2) for i in range(1,5)]
    return specs

def filter_paths(paths : dict, spec : str) -> dict:
    """
    Filters the paths according to the given attribute/specification
    """
    all_paths = defaultdict(dict)
    for mag in paths.keys():
        specs = get_specs(mag, spec)
        paths_restructured = defaultdict(list)
        for path in paths[mag]:
            for s in specs:
                if s in path:
                    paths_restructured[s].append(path)
        all_paths[mag] = paths_restructured
    return all_paths


def get_stats(filtered_paths : dict) -> dict:
    """
    Counts the pixel values of each image to give a value 
    distribution that will be plotted
    """
    stats_dict = defaultdict(dict)
    for mag in filtered_paths.keys():
        for spec,paths in filtered_paths[mag].items():
            counter = Counter()
            for path in paths:
                img = cv2.imread(path, -1)
                count = Counter(list(img.ravel()))
                counter += count
            stats_dict[mag][spec] = counter
        print(mag)
    return stats_dict
