import concurrent
import copy
import json
import math
import os
import random
import shutil
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Union, Sized
)

import cv2
import numpy as np
from tqdm import tqdm

__all__ = [
    "OCRDatasetV2",
    "OCRCLSDatasetV2",
    "OCRRECDatasetV2",
    "dump_ocr_dataset",
    "matting_ocr_dataset",
    "split_ocr_dataset",
    "YOLODataset",
    "split",
    "validate_normalized_coords",
    "dump_yolo_dataset",
    "VOCDataset",
]

from AITools.base.dataset_def import IterableDataset, T_co
from AITools.base.vision_def import IMG_FORMATS
from AITools.core.manager import ComponentManager
from AITools.comp.functions import (
    parse_ppocr_label, imread, imwrite, plot_box_and_text_v2, img2label_path
)
from AITools.comp.parser import XMLParser

DATASETS = ComponentManager("datasets")


def _parse_slice(s: slice, obj: Sized) -> list:
    """Converts slice objects to a list of valid indexes"""
    start, stop, step = s.indices(len(obj))
    return list(range(start, stop, step))


def _validate_indices(indices, obj: Sized) -> list:
    """Validate and filter the index list"""
    max_idx = len(obj) - 1
    return [idx for idx in indices if 0 <= idx <= max_idx]


class DoNotReadImage:
    """A context manager for disabling image reading"""

    def __init__(self, d):
        self.prev = d
        self.record_read_image = d.is_read_image

    def __enter__(self):
        self.record_read_image = self.prev.is_read_image
        self.prev._read_image = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.prev._read_image = self.record_read_image
        return False

    def __call__(self, func):
        """作为装饰器使用"""

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


@DATASETS.register_component
class OCRDatasetV2(IterableDataset):
    def __init__(
        self,
        root: Union[str, List[str]] = None,
        with_image: bool = True,
        with_label: bool = False,
        *,
        read_image: bool = False,
        transformers=None,
        subject_to="image",
        label_file="Label.txt",
        **kwargs
    ):
        """
        Args:
            root (str or list of str): The root directory of the dataset.
            with_image (bool): Whether to load the image.
            with_label (bool): Whether to load the label.
            read_image (bool): Whether to read the image.
            transformers (list of callable): The transformers to apply to the data.
            subject_to (str): The subject to which the dataset is subject to, must be 'image' or 'label'.
            Only useful if with_label is true.
            label_file (str): The label file name.

        """
        super().__init__()
        self._image_map = {}
        self._label_map = {}
        self._place_map = {}
        self._roots_map = {}
        self._lab_files = {}
        self._index = 0
        self._begin = 0
        self._with_image = with_image
        self._with_label = with_label
        self._read_image = read_image
        self.transformers = transformers
        self._roots_set = set()  # New: Quick directory existence check
        self._dir_basename_map = defaultdict(set)  # New: Mapping directory to file name
        if subject_to not in ["image", "label"]:
            raise ValueError("subject_to must be 'image' or 'label'.")
        self.subject_to = subject_to

        self._parser_root(root, with_image, with_label, label_file)
        _start = len(self._image_map)
        for idx, _image_root in self._roots_map.items():
            self._parse(_image_root, idx, _start)
            _end = len(self._image_map)
            for i in range(_start, _end):
                self._place_map[i] = idx
            _start = _end

    def _parser_root(self, root, with_image, with_label, label_file):
        if with_image and not with_label:
            if isinstance(root, list):
                self._roots_map = {idx: root for idx, root in enumerate(root)}
            elif isinstance(root, str):
                self._roots_map = {0: root}
            elif root is None:
                self._with_image = False
            else:
                raise ValueError("Not expected type of root: {}".format(type(root)))
        elif with_image and with_label:
            if isinstance(root, list):
                self._roots_map = {idx: root for idx, root in enumerate(root)}
                if os.path.isabs(label_file):
                    raise ValueError("label_file must be relative path when root is list")
                self._lab_files = {idx: os.path.join(root, label_file) for idx, root in self._roots_map.items()}
            elif isinstance(root, str):
                self._roots_map[0] = root
                self._lab_files[0] = label_file if os.path.isabs(label_file) else os.path.join(root, label_file)
            else:
                raise ValueError("root must be list or str when 'with_image=True' and 'with_label=True', "
                                 "unexpected: {}".format(root))
        elif not with_image and not with_label:
            if isinstance(root, str):
                os.makedirs(root, exist_ok=True)
                self._roots_map[0] = root
                label_file = label_file if os.path.isabs(label_file) else os.path.join(root, label_file)
                if not os.path.exists(label_file):
                    open(label_file, "w", encoding="utf-8").close()
                self._lab_files[0] = label_file
            elif root is None:
                pass
            else:
                raise ValueError("When 'with_image=False' and 'with_label=False', the root must be a path or None; "
                                 "If root is a path and doesn't exist, it will be created automatically, if root "
                                 "is None, an empty dataset object is declared.")
        else:
            raise ValueError("The case of only labels without images is not supported.")

    def _parse(self, image_root, root_idx=None, offset=0, label_file_encoding='utf-8'):
        abs_image_root = os.path.abspath(os.path.normpath(image_root))
        if not os.path.isdir(abs_image_root):
            raise ValueError(f"image_path must be a directory: {abs_image_root}.")
        if not os.path.exists(abs_image_root):
            raise FileNotFoundError(f"Image path {abs_image_root} does not exist.")

        if abs_image_root not in self._roots_set:
            if root_idx is None:
                root_idx = len(self._roots_map)
                self._roots_map[root_idx] = abs_image_root
            self._roots_set.add(abs_image_root)
            self._dir_basename_map[abs_image_root] = set()
        else:
            root_idx = [k for k, v in self._roots_map.items() if v == abs_image_root][0]

        label_file = self._lab_files.get(root_idx, None)
        data = {} if label_file is None else self._parse_file(label_file, encoding=label_file_encoding)
        if self.subject_to == "label" and self.with_label:
            for i, (im, la) in enumerate(data.items()):
                idx = offset + i
                img_basename = im.replace(abs_image_root + f"{os.sep}", "")
                self._image_map[idx] = img_basename
                self._place_map[idx] = root_idx
                self._dir_basename_map[abs_image_root].add(img_basename)
                self._label_map[idx] = la
        else:
            valid_images = [p for p in os.listdir(abs_image_root) if p.endswith(tuple(IMG_FORMATS))]
            for i, im in enumerate(valid_images):
                idx = offset + i
                self._image_map[idx] = im
                self._place_map[idx] = root_idx
                self._dir_basename_map[abs_image_root].add(im)
                if self.with_label:
                    # TODO: Strictly filter images that are not in the label
                    self._label_map[idx] = data.get(os.path.join(abs_image_root, im), [])

    def _parse_label(self, contents):
        return json.loads(self.fmt_label_loads(contents))

    def _parse_file(self, file, encoding='utf-8'):
        """
        Regulations：The path in the annotated file must be an absolute path or a
        relative path starting with the folder name of the current annotated file
        """
        data = {}
        file_dirname = os.path.dirname(file)
        with open(file, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                p, a = os.path.normpath(parts[0]), parts[1]
                image_name = p if os.path.isabs(p) else os.path.join(file_dirname, *p.split(os.sep)[1:])
                if os.path.exists(image_name):
                    data[image_name] = self._parse_label(a)
        return data

    @classmethod
    def fmt_label_dumps(cls, label):
        return (str(label).replace("'", '"')
                .replace('"difficult": 0', '"difficult": false')
                .replace('"difficult": 1', '"difficult": true'))

    @classmethod
    def fmt_label_loads(cls, label: str):
        return (label.strip("\n").strip("\r")
                .replace('"difficult": false', '"difficult": 0')
                .replace('"difficult": true', '"difficult": 1')
                .replace('"difficult": False', '"difficult": 0')
                .replace('"difficult": True', '"difficult": 1'))

    def __iter__(self) -> Iterator[T_co]:
        self._index = self._begin
        return self

    def __next__(self):
        if self._index < len(self):
            ret = self[self._index]
            self._index += 1
            return ret
        else:
            self._index = self._begin
            raise StopIteration(f"Iterator out of range, stop.")

    def __len__(self):
        return len(self._image_map)

    def __getitem__(self, index):
        """
        Enhanced index access, supported in the following forms:
        - dataset[5]        -> Single sample
        - dataset[1:10:2]   -> Slicing generates a new data set
        - dataset[[1,3,5]]  -> The list index generates a new data set

        """
        if isinstance(index, slice):
            return self.subset(_parse_slice(index, self))

        elif isinstance(index, (list, tuple, np.ndarray)):
            return self.subset(_validate_indices(index, self))

        try:
            return self._get_data_item(index) if self._read_image else self._get_single_item(index)
        except KeyError:
            raise IndexError(f"index {index} out of dataset range [0, {len(self) -1}]")

    def _get_single_item(self, index):
        """Basic method of obtaining a single sample"""
        if self._begin <= index < len(self):
            root_dir = self._roots_map[self._place_map[index]]
            img_name = self._image_map[index]
            full_path = os.path.join(root_dir, img_name)

            if self.with_label:
                return full_path, self._label_map.get(index, None)
            return full_path, None

        raise IndexError(f"index {index} is out of the valid range")

    def _get_data_item(self, index):
        image_path, anno_label = self._get_single_item(index)
        return parse_ppocr_label(image_path, anno_label)

    def _update_directory_mappings(self):
        """Update the directory mapping relationship of the subset"""
        # rebuild roots_map
        active_roots = set(self._place_map.values())
        self._roots_map = {rid: path for rid, path in self._roots_map.items() if rid in active_roots}

        # rebuild root_dirs
        self._roots_set = {path for path in self._roots_map.values()}

        # rebuild dir_basename_map
        self._dir_basename_map = defaultdict(set)
        for idx in self._image_map:
            root_id = self._place_map[idx]
            root_path = self._roots_map[root_id]
            self._dir_basename_map[root_path].add(self._image_map[idx])

    def __setitem__(self, index: int, data: Union[str, tuple]):
        """
        Support two assignment modes:
        1. Only update the image path (when with_label=False)
           example: dataset[0] = "/new/path/img1.jpg"
        2. Update the image path and label simultaneously (when with_label=True)
           example: dataset[1] = ("/new/path/img2.jpg", {"text": "Hello"})
        """
        # ==================== parameter checking ====================
        if index < 0 or index >= len(self):
            raise IndexError(f"index {index} out of dataset range [0, {len(self) - 1}].")

        if self.with_label:
            if not isinstance(data, tuple) or len(data) != 2:
                raise ValueError("The (path, label) tuple is required.")
            new_image_path, new_label = data
        else:
            if not isinstance(data, str):
                raise ValueError("need to provide an image path string.")
            new_image_path = data
            new_label = None

        # ==================== multithread processing ====================
        # New Path of Standardization
        abs_new_path = os.path.abspath(os.path.normpath(new_image_path))
        new_dir = os.path.dirname(abs_new_path)
        new_basename = os.path.basename(abs_new_path)

        # Obtain the original data information
        old_basename = self._image_map[index]
        old_root_id = self._place_map[index]
        old_dir = self._roots_map[old_root_id]

        # ==================== Path conflict checking ====================
        # Check whether the new file name already exists in the target directory
        if new_dir == old_dir:
            # Update the file names in the same directory
            if new_basename != old_basename and new_basename in self._dir_basename_map[old_dir]:
                raise FileExistsError(f"File {new_basename} already exists in directory {old_dir}.")
            new_root_id = old_root_id
        else:
            # Cross-directory updates require checking the new directory
            if new_dir in self._roots_set and new_basename in self._dir_basename_map[new_dir]:
                raise FileExistsError(f"File {new_basename} already exists in directory {new_dir}")

            new_root_id = len(self._roots_map)
            self._roots_map[new_root_id] = new_dir
            self._roots_set.add(new_dir)
            self._dir_basename_map[new_dir] = set()

        # ==================== Update the data structure ====================
        # 1. Clean up the old records
        self._dir_basename_map[old_dir].discard(old_basename)

        # 2. Update the core mapping
        self._image_map[index] = new_basename
        self._place_map[index] = new_root_id
        self._dir_basename_map[new_dir].add(new_basename)

        # 3. Update the label (if enabled)
        if self.with_label:
            self._label_map[index] = new_label

        # 4. Automatically clean up empty directories
        if not self._dir_basename_map[old_dir]:
            del self._roots_map[old_root_id]
            self._roots_set.discard(old_dir)
            del self._dir_basename_map[old_dir]
            print(f"The directory {old_dir} has no data and is automatically removed")

    def __add__(self, other: "OCRDatasetV2") -> "OCRDatasetV2":
        """
        Merge two data sets, overwriting the current data with the other data set when there is data with the same path
        Returns a new dataset object, the original dataset will not be modified

        :param other: Another data set to merge
        :return: The new data set after the merger
        """
        if not isinstance(other, OCRDatasetV2):
            raise TypeError("Only OCRDatasetV2 objects can be merged.")

        if self.with_image != other.with_image or self.with_label != other.with_label:
            raise ValueError("Data set configuration incompatibility (with_image/with_label inconsistency).")

        merged_dataset = self.copy()

        # Create a mapping between the directory path and root_id (new root_id after merging)
        root_path_to_id = {v: k for k, v in merged_dataset._roots_map.items()}

        # Merge the root directory of 'other'
        for root_id, root_path in other._roots_map.items():
            if root_path not in root_path_to_id:
                # Add a new root directory
                new_root_id = len(merged_dataset._roots_map)
                merged_dataset._roots_map[new_root_id] = root_path
                merged_dataset._roots_set.add(root_path)
                merged_dataset._dir_basename_map[root_path] = set()
                root_path_to_id[root_path] = new_root_id

        # Merge data item
        # Create a quick path lookup table {abs path: index}
        existing_paths = {
            os.path.join(merged_dataset._roots_map[rid], img): idx
            for idx, (rid, img) in enumerate(zip(merged_dataset._place_map.values(),
                                                 merged_dataset._image_map.values()))
        }

        # Iterate over the 'other' dataset, retrieving data items
        for other_idx in range(len(other)):
            other_item = other[other_idx]
            abs_path = os.path.abspath(
                other_item[0] if isinstance(other_item, tuple) else other_item
            )
            dir_path = other.directory(other_idx)
            base_name = other.image(other_idx)

            # Conflict handling logic
            if abs_path in existing_paths:
                # Overlay existing data
                merged_idx = existing_paths[abs_path]

                merged_dataset._image_map[merged_idx] = base_name
                merged_dataset._place_map[merged_idx] = root_path_to_id[dir_path]

                if merged_dataset._with_label:
                    merged_dataset._label_map[merged_idx] = other._label_map.get(other_idx, [])

                merged_dataset._dir_basename_map[dir_path].add(base_name)
            else:
                # New data item
                new_idx = len(merged_dataset._image_map)

                merged_dataset._image_map[new_idx] = base_name
                merged_dataset._place_map[new_idx] = root_path_to_id[dir_path]

                if merged_dataset._with_label:
                    merged_dataset._label_map[new_idx] = other._label_map.get(other_idx, [])

                merged_dataset._dir_basename_map[dir_path].add(base_name)
                existing_paths[abs_path] = new_idx

        # Merge label file
        if self.with_image and self.with_label:
            # other overwrite current
            merged_dataset._lab_files.update(other._lab_files)

        return merged_dataset

    @property
    def length(self):
        return len(self)

    @property
    def images(self):
        """
        All data absolute paths.
        Returns: list of str
        """
        return [os.path.join(self._roots_map[self._place_map[idx]], name)
                for idx, name in self._image_map.items()]

    @property
    def image_indexes(self):
        return list(self._image_map.keys())

    @property
    def image_map(self):
        return self._image_map

    @property
    def directories(self):
        return list(self._roots_map.values())

    @property
    def label_files(self):
        return list(self._lab_files.values())

    @property
    def with_image(self):
        return self._with_image

    @property
    def with_label(self):
        return self._with_label

    @with_image.setter
    def with_image(self, value: bool):
        self._with_image = value

    @with_label.setter
    def with_label(self, value: bool):
        self._with_label = value

    def image(self, index: int):
        return self._image_map[index]

    def directory(self, index: int):
        return self._roots_map[self._place_map[index]]

    @property
    def roots_map(self):
        return self._roots_map

    def label(self, index: int):
        return self._label_map[index] if self.with_label else None

    def subset(self, indices: list) -> "OCRDatasetV2":
        """Create a subdataset based on the index list"""
        # Create a new instance but skip the initialization process
        new_dataset = self.__class__.__new__(self.__class__)
        new_dataset.__dict__ = copy.deepcopy(self.__dict__)

        # Update the core data map
        new_dataset._image_map = {}
        new_dataset._place_map = {}
        new_dataset._label_map = {} if self.with_label else None
        new_dataset.aug_map = {}

        # Populate the filtered data
        for new_idx, old_idx in enumerate(indices):
            new_dataset._image_map[new_idx] = self._image_map[old_idx]
            new_dataset._place_map[new_idx] = self._place_map[old_idx]
            if self.with_label:
                new_dataset._label_map[new_idx] = self._label_map.get(old_idx, {})

        # Update directory mapping
        new_dataset._update_directory_mappings()
        return new_dataset

    def append(self, image_path, label=None):
        abs_image_path = os.path.abspath(os.path.normpath(image_path))
        if not os.path.exists(abs_image_path) or not abs_image_path.endswith(tuple(IMG_FORMATS)):
            raise ValueError(f"File '{abs_image_path}' does not exist or is not an image file.")
        dirname = os.path.dirname(abs_image_path)
        basename = os.path.basename(abs_image_path)
        if dirname in self._roots_set and basename in self._dir_basename_map[dirname]:
            raise ValueError(f"File '{basename}' already exists in directory '{dirname}'.")

        if not self._with_image and not self._with_label:
            self._with_image = True
            if label is not None:
                self._with_label = True

        idx = len(self._image_map)
        if self.with_label:
            if label is None:
                raise ValueError("The current configuration requires that label be provided.")
            self._label_map[idx] = label
        else:
            if label is not None:
                warnings.warn("The current configuration does not support labels. The entered labels are ignored.")
        if dirname in self._roots_set:
            root_idx = [k for k, v in self._roots_map.items() if v == dirname][0]
        else:
            root_idx = len(self._roots_map)
            self._roots_map[root_idx] = dirname
            self._roots_set.add(dirname)
            self._dir_basename_map[dirname] = set()
        self._image_map[idx] = basename
        self._place_map[idx] = root_idx
        self._dir_basename_map[dirname].add(basename)

    def split(self, ratio: Union[float, List[float]] = None, subset_name=None, seed: int = None, grouped: bool = True, specified=None):
        return split(self, ratio=ratio, subset_name=subset_name, seed=seed, grouped=grouped, specified=specified)

    def wash(self, image_list: List[Union[str, int]], mode='drop') -> 'OCRDatasetV2':
        """
        According to the input image_list, discard the unwanted data, update self._image_map, self._label_map,
        self._place_map, self._roots_map, self._lab_files and other information.
        :param image_list: a list of image paths or indices
        :param mode: 'drop' or 'keep', if 'keep', the image_list will be kept, otherwise, the image_list will be dropped
        """
        image_set = set()
        for p in image_list:
            if isinstance(p, int):
                if p not in self._image_map:
                    raise ValueError(f"Invalid image index: {p}")
                root_idx = self._place_map[p]
                root_dir = self._roots_map[root_idx]
                img_name = self._image_map[p]
                abs_path = os.path.abspath(os.path.join(root_dir, img_name))
                image_set.add(abs_path)
            elif isinstance(p, str):
                abs_path = os.path.abspath(os.path.normpath(p))
                image_set.add(abs_path)
            else:
                raise TypeError("The image_list element must be a string or integer.")

        if mode == 'drop':
            def is_to_keep(x):
                return x not in image_set
        elif mode == 'keep':
            def is_to_keep(x):
                return x in image_set
        else:
            raise ValueError("mode must be 'drop' or 'keep'.")

        retained = []
        for old_index in self._image_map.keys():
            root_idx = self._place_map[old_index]
            root_dir = self._roots_map[root_idx]
            img_name = self._image_map[old_index]
            abs_path = os.path.abspath(os.path.normpath(os.path.join(root_dir, img_name)))
            if is_to_keep(abs_path):
                label = self._label_map.get(old_index, []) if self.with_label else []
                retained.append((root_dir, img_name, label))

        # Build a new _roots_map, preserving the order and removing the weight
        seen_roots = set()
        unique_root_dirs = []
        for rd, _, _ in retained:
            if rd not in seen_roots:
                seen_roots.add(rd)
                unique_root_dirs.append(rd)
        new_roots_map = {idx: rd for idx, rd in enumerate(unique_root_dirs)}

        # Build new _image_map, _label_map, _place_map
        new_image_map = {}
        new_label_map = {}
        new_place_map = {}
        for new_index, (rd, img_name, label) in enumerate(retained):
            new_image_map[new_index] = img_name
            if self.with_label:
                new_label_map[new_index] = label
            # Find the corresponding new root_idx
            new_root_idx = [k for k, v in new_roots_map.items() if v == rd][0]
            new_place_map[new_index] = new_root_idx

        # Build a new self._lab_files, keeping only the entries corresponding to the existing root_dir
        new_lab_files = {}
        if self.with_image and self.with_label:
            for old_root_idx, lab_file in self._lab_files.items():
                old_rd = self._roots_map.get(old_root_idx)
                if old_rd in new_roots_map.values():
                    new_root_idx = [k for k, v in new_roots_map.items() if v == old_rd][0]
                    new_lab_files[new_root_idx] = lab_file

        self._image_map = new_image_map
        self._label_map = new_label_map if self.with_label else {}
        self._place_map = new_place_map
        self._roots_map = new_roots_map
        self._lab_files = new_lab_files if (self.with_image and self.with_label) else {}
        self._roots_set = set(new_roots_map.values())
        self._dir_basename_map = defaultdict(set)
        for new_index in new_image_map.keys():
            root_dir = new_roots_map[new_place_map[new_index]]
            img_name = new_image_map[new_index]
            self._dir_basename_map[root_dir].add(img_name)

        return self

    def copy(self) -> Any:
        """Creates a deep-copy copy of the current dataset"""
        new = OCRDatasetV2.__new__(OCRDatasetV2)  # Avoid calling __init__

        new._image_map = copy.deepcopy(self._image_map)
        new._label_map = copy.deepcopy(self._label_map)
        new._place_map = copy.deepcopy(self._place_map)
        new._roots_map = copy.deepcopy(self._roots_map)
        new._lab_files = copy.deepcopy(self._lab_files)
        new._roots_set = copy.deepcopy(self._roots_set)
        new._dir_basename_map = copy.deepcopy(self._dir_basename_map)
        new._index = self._index
        new._begin = self._begin

        new._with_image = self._with_image
        new._with_label = self._with_label
        new._read_image = self._read_image
        new.subject_to = self.subject_to
        new.transformers = copy.deepcopy(self.transformers)

        return new

    def sample(
        self,
        ratio: Union[float, Dict[Union[int, str], float]],
        condition: Callable = None,
        seed: int = None,
        parallel: bool = False
    ) -> list[int]:
        """
        Stratified sampling by folder

        :param ratio: Sample scale, which can be global scale or dictionary {root directory ID/ path: scale}
        :param condition: Filter function `condition(item) -> bool`
        :param seed: Random seed (full reproducibility not guaranteed in parallel mode)
        :param parallel: Enable parallel processing (for large-scale data)
        :return: Sampling index list
        """
        # parameter checking
        if not 0 <= (ratio if isinstance(ratio, float) else max(ratio.values())) <= 1:
            raise ValueError("The sampling ratio must be between [0.0 and 1.0].")
        if condition is not None:
            if not callable(condition):
                raise TypeError("condition must be a callable function.")
        else:
            def condition(*args): return True

        # Seed setting
        if seed is not None:
            random.seed(seed)
            if parallel:
                warnings.warn(f"Sample seed set: {seed} (may not be fully reproducible in parallel mode)")

        # Dynamic proportional preprocessing
        ratio_map = self._parse_ratio(ratio)

        # Group preprocessing
        root_groups = self.get_root_groups()

        # Parallel/Serial processing
        if parallel:
            return self._parallel_sample(root_groups, ratio_map, condition)
        else:
            return self._sequential_sample(root_groups, ratio_map, condition)

    def _parse_ratio(self, ratio: Union[float, Dict]) -> Dict[int, float]:
        """Converts ratio uniformly to the format {root directory ID: ratio}"""
        if isinstance(ratio, float):
            return {rid: ratio for rid in self._roots_map}

        resolved_ratio = {}
        for k, v in ratio.items():
            if isinstance(k, str):
                abs_path = os.path.abspath(k)
                found = [rid for rid, path in self._roots_map.items() if path == abs_path]
                if not found:
                    raise KeyError(f"The root directory for the path was not found: {k}")
                resolved_ratio[found[0]] = v
            else:
                if k not in self._roots_map:
                    raise KeyError(f"Invalid root directory id: {k}")
                resolved_ratio[k] = v
        return resolved_ratio

    def _process_root_group(self, rid: int, indices: list, ratio: float, condition: Callable) -> list:
        """Handles sampling logic for a single root directory"""
        valid_indices = [i for i in indices if condition(self[i])]
        if not valid_indices:
            return []

        sample_size = max(1, math.ceil(len(valid_indices) * ratio))
        sample_size = min(sample_size, len(valid_indices))

        return random.sample(valid_indices, sample_size) if sample_size < len(valid_indices) else valid_indices

    def _sequential_sample(self, groups: Dict, ratio_map: Dict, condition: Callable) -> list:
        """Serial sampling"""
        subset = []
        for rid, indices in groups.items():
            ratio = ratio_map.get(rid, 0.0)  # A directory with no specified ratio is not sampled by default
            subset.extend(self._process_root_group(rid, indices, ratio, condition))
        return subset

    def _parallel_sample(self, groups: Dict, ratio_map: Dict, condition: Callable) -> list:
        """Parallel sampling (for IO-intensive tasks)"""
        subset = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for rid, indices in groups.items():
                ratio = ratio_map.get(rid, 0.0)
                futures.append(
                    executor.submit(
                        self._process_root_group,
                        rid, indices, ratio, condition
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                subset.extend(future.result())
        return subset

    def get_root_groups(self):
        """Gets an index of samples grouped by root directory"""
        groups = defaultdict(list)
        for idx in self._image_map:
            root_id = self._place_map[idx]
            groups[root_id].append(idx)
        return groups


@DATASETS.register_component
class OCRRECDatasetV2(OCRDatasetV2):
    def __init__(self, *args, **kwargs):
        subject_to = kwargs.pop("subject_to", "label")
        super().__init__(*args, subject_to=subject_to, **kwargs)

    def _parse_label(self, contents):
        return self.fmt_label_loads(contents)

    @classmethod
    def fmt_label_dumps(cls, label: str):
        return label.strip("\n").strip("\r")

    @classmethod
    def fmt_label_loads(cls, label: str):
        return label.strip("\n").strip("\r")


@DATASETS.register_component
class OCRCLSDatasetV2(OCRDatasetV2):
    def __init__(self, *args, categories: dict[int, str], **kwargs):
        self._categories_i2s = categories
        self._categories_s2i = {v: k for k, v in self._categories_i2s.items()}
        super().__init__(*args, **kwargs)

    def _parse_label(self, contents):
        return self.fmt_label_loads(contents)

    def fmt_label_dumps(self, label: int):
        if label not in self._categories_i2s:
            raise ValueError(f"Unknown label(int): '{label}'")
        return self._categories_i2s[label]

    def fmt_label_loads(self, label: str):
        label = label.strip("\n").strip("\r")
        if label not in self._categories_s2i:
            raise ValueError(f"Unknown label(str): '{label}'")
        return self._categories_s2i[label]

    def copy(self):
        new = super().copy()
        new._categories_i2s = copy.deepcopy(self._categories_i2s)
        new._categories_s2i = copy.deepcopy(self._categories_s2i)
        return new

    def category(self, l: Union[int, str]):
        if isinstance(l, int):
            return self._categories_i2s[l]
        elif isinstance(l, str):
            return self._categories_s2i[l]
        else:
            raise TypeError(f"Unsupported label type: {type(l)}")


def dump_ocr_dataset(
    dataset: OCRDatasetV2,
    destination: str,
    image_file_op: str = "copy",
    custom_image_label_op: Callable = None,
    label_file_name: str = "Label.txt",
    label_file_encoding="utf-8",
    overwriting: bool = False,
    tqdm_enable: bool = True
):
    """
    Dump dataset to a new directory.

    @param dataset: OCRDatasetV2
    @param destination: Destination directory
    @param image_file_op: Operation on image files, "copy" or "move"
    @param custom_image_label_op: Custom image file operation
    @param label_file_name: Label file name
    @param label_file_encoding: Label file encoding
    @param overwriting: Whether to overwrite the destination directory
    @param tqdm_enable: Whether to enable tqdm progress bar
    """
    if not dataset:
        raise ValueError("Dataset is empty.")
    if custom_image_label_op is None:
        if image_file_op == "copy":
            img_op = shutil.copy
        elif image_file_op == "move":
            img_op = shutil.move

        def image_label_op(_dst_dir, _img_path: str, _label_data=None, _label_file=None, _label_op=None, **kwargs):
            basename = os.path.basename(_img_path)
            img_op(_img_path, os.path.join(_dst_dir, basename))
            if _label_file is not None and _label_data is not None:
                dirname = os.path.basename(_dst_dir)
                _label_str = _label_op(_label_data)
                _label_file.write(f"{dirname}/{basename}\t{_label_str}\n")

    elif callable(custom_image_label_op):
        image_label_op = custom_image_label_op
    else:
        raise ValueError(f"Unsupported image file operation: '{image_file_op}'")

    if os.path.exists(destination):
        if os.listdir(destination) and not overwriting:
            raise FileExistsError(f"Destination path {destination} is not empty")
    else:
        os.makedirs(destination, exist_ok=True)

    commonpath = os.path.commonpath(list(dataset.roots_map.values()))

    opened_files = {}
    try:
        iterator = tqdm(dataset, desc=f"Dumping dataset") if tqdm_enable else dataset
        if dataset.with_label:
            # Unified processing for labeled data
            for i, (image_path, label_data) in enumerate(iterator):
                # Determine destination directory
                if len(dataset.directories) > 1:
                    src_dir = os.path.dirname(image_path)
                    # dir_name = os.path.basename(src_dir)
                    # dst_dir = os.path.join(destination, dir_name)
                    dst_dir = os.path.join(destination, os.path.relpath(src_dir, commonpath))
                else:
                    dst_dir = destination

                os.makedirs(dst_dir, exist_ok=True)
                label_file = os.path.join(dst_dir, label_file_name)
                if label_file not in opened_files:
                    opened_files[label_file] = open(label_file, "w", encoding=label_file_encoding)
                image_label_op(dst_dir, image_path, label_data, opened_files[label_file], dataset.fmt_label_dumps,
                               index=i)
        else:
            # Unified processing for unlabeled data
            for img_path in iterator:
                if len(dataset.directories) > 1:
                    src_dir = os.path.dirname(img_path)
                    dir_name = os.path.basename(src_dir)
                    dst_dir = os.path.join(destination, dir_name)
                else:
                    dst_dir = destination

                os.makedirs(dst_dir, exist_ok=True)
                image_label_op(dst_dir, img_path)
    finally:
        for file_handle in opened_files.values():
            file_handle.close()


def split_ocr_dataset(src_dir: str, dst_dir: str, set_type='det', path_collect_func=None, seed=None, subset_name=None,
                      ratio=None, overwriting=False, label_file_name="Label.txt", label_file_encoding="utf-8",
                      parallel: bool = False, normative: bool = True):
    if set_type == 'det':
        DatasetClass = OCRDatasetV2
    elif set_type == 'rec':
        DatasetClass = OCRRECDatasetV2
    elif set_type == 'cls':
        DatasetClass = OCRCLSDatasetV2
    else:
        raise ValueError(f"Unsupported set_type: '{set_type}'")
    if path_collect_func is None:
        def path_collect_func(p):
            return [os.path.join(p, i) for i in os.listdir(p) if os.path.isdir(os.path.join(p, i))]
    else:
        if not callable(path_collect_func):
            raise ValueError(f"Unsupported path_collect_func: '{path_collect_func}'")
    if ratio is None:
        ratio = [0.5, 0.2, 0.3]
    elif isinstance(ratio, float):
        ratio = [ratio, 1 - ratio]
    if subset_name is None:
        subset_name = ['train', 'val', 'test']
    if len(ratio) > len(subset_name):
        raise ValueError("The length of subset_name and ratio must be equal")
    datasets = []
    dataset_split = {subset_name[i]: [] for i, s_name in enumerate(ratio)}

    def _wash_and_dump(_rid, _path, _dataset, _set_idxes, _set_name, _label_file_name, _label_file_encoding):
        _dst_dir_name = os.path.basename(_path)
        _dst_sub_dir = os.path.join(dst_dir, _set_name, _dst_dir_name)
        try:
            new = _dataset.copy()
            new.wash(_set_idxes, 'keep')
            dump_ocr_dataset(new, _dst_sub_dir,
                             label_file_name=_label_file_name,
                             label_file_encoding=_label_file_encoding,
                             overwriting=overwriting,
                             tqdm_enable=False)
        except Exception as e:
            print(f"Error occurred while processing subset {_rid}: {e}")
            return False, _rid
        return True, _rid

    def _process(_rid, _path, _dataset, _set_idxes, _set_name, _label_file_name, _label_file_encoding):
        _dst_dir_name = os.path.basename(_path)
        _dst_sub_dir = os.path.join(dst_dir, _set_name, _dst_dir_name)
        if not os.path.exists(_dst_sub_dir):
            os.makedirs(_dst_sub_dir, exist_ok=True)
        else:
            if not overwriting:
                raise FileExistsError(f"Directory '{_dst_sub_dir}' already exists")
        with open(os.path.join(_dst_sub_dir, _label_file_name), "w", encoding=_label_file_encoding) as f:
            for idx in _set_idxes:
                im, la = _dataset[idx]
                _im_basename = os.path.basename(im)
                _la_dump_str = _dataset.fmt_label_dumps(la)
                shutil.copy(im, _dst_sub_dir)
                f.write("{}/{}\t{}\n".format(_dst_dir_name, _im_basename, _la_dump_str))
        dataset_split[_set_name].append(_set_idxes)

    def _slowly_dump(_path, _subset, _dataset):
        """ Slow """
        for _rid, (_set_name, _set_idxes) in enumerate(_subset.items()):
            _process(_rid, _path, _dataset, _set_idxes, _set_name, label_file_name, label_file_encoding)

    def _normatively_dump(_path, _subset, _dataset):
        """ Fast and safe """
        with ThreadPoolExecutor() as executor:
            futures = []
            for _rid, (_set_name, _set_idxes) in enumerate(_subset.items()):
                futures.append(
                    executor.submit(
                        _wash_and_dump,
                        _rid, _path, _dataset, _set_idxes, _set_name, label_file_name, label_file_encoding
                    )
                )
                dataset_split[_set_name].append(_set_idxes)

            for future in concurrent.futures.as_completed(futures):
                success, rid = future.result()

    def _quickly_dump(_path, _subset, _dataset):
        """ Fastest but not safe """
        with ThreadPoolExecutor() as executor:
            futures = []
            for _rid, (_set_name, _set_idxes) in enumerate(_subset.items()):
                futures.append(
                    executor.submit(
                        _process,
                        _rid, _path, _dataset, _set_idxes, _set_name, label_file_name, label_file_encoding
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                future.result()

    process_func = _slowly_dump if not parallel else _normatively_dump if normative else _quickly_dump
    for i, path in enumerate(tqdm(path_collect_func(src_dir), desc="Splitting dataset")):
        dataset = DatasetClass(path, with_label=True)
        subsets = dataset.split(seed=seed, ratio=ratio, subset_name=subset_name)
        process_func(path, subsets, dataset)
        datasets.append(dataset)

    for n in subset_name:
        set_dir = os.path.join(dst_dir, n)
        rec_label_files = [os.path.join(set_dir, p, label_file_name)
                           for p in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, p))]
        dst_rec_label_file = os.path.join(set_dir, label_file_name)
        union_ocr_dataset_label(rec_label_files, dst_rec_label_file)

    return datasets, dataset_split


def matting_ocr_dataset(dataset: OCRDatasetV2, output: str, label_file_name: str = "Label.txt",
                        label_file_encoding: str = "utf-8", overwriting: bool = False):
    def write(_save_name, _roi, _str, show=False):
        imwrite(os.path.join(output, _save_name), _roi)
        f.write(f"{os.path.basename(output)}/{_save_name}\t{_str}\n")
        if show:
            plot_box_and_text_v2(roi, [], _str, text_lw_scale=0.3, text_color=(0, 255, 0))

    os.makedirs(output, exist_ok=overwriting)
    if dataset.with_label:
        replace_char = ["\n", "\r", "\\", "/", ":", "*", "?", "<", ">", "|", "'", '"']
        with open(os.path.join(output, label_file_name), "w", encoding=label_file_encoding) as f:
            for i, (img_path, labels) in enumerate(tqdm(dataset, desc=f"Matting")):
                basename = os.path.basename(img_path)
                filename, suffix = basename.rsplit(".", 1)
                img = imread(img_path)

                try:
                    if len(labels) == 1:
                        rect = cv2.boundingRect(np.array(labels[0]["points"]))
                        if len(img.shape) == 3:
                            roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]
                            save_name = f"{filename}.{suffix}"
                            write(save_name, roi, labels[0]["transcription"])
                    else:
                        for j, l in enumerate(labels):
                            rect = cv2.boundingRect(np.array(l["points"]))
                            if len(img.shape) == 3:
                                roi = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :]
                                for c in replace_char:
                                    l_str = l["transcription"].replace(c, "")
                                save_name = f"{filename}_{j}_{l_str}.{suffix}"
                                write(save_name, roi, l["transcription"])
                except Exception as e:
                    print(e, img_path, labels if labels is not None else "label is none")


def union_ocr_dataset_label(label_files, dst_file):
    dst_dirname = os.path.basename(os.path.dirname(dst_file))
    with open(dst_file, "w", encoding="utf-8") as f:
        for label_file in label_files:
            with open(label_file, "r", encoding="utf-8") as f1:
                for line in f1:
                    f.write(f"{dst_dirname}/" + line)


class SeparateDataset(IterableDataset):

    def __init__(
        self,
        root: Union[str, Path, List[Union[str, Path]]] = None,
        *,
        with_image: bool = True,
        with_label: bool = True,
        image_dirname: str = "images",
        label_dirname: str = "labels",
        task: str = "det",
        categories: Union[Dict[str, int], Dict[int, str]] = None,
        subject_to: str = "image",
        read_image: bool = False,
        transformers=None,
        kpt_shape=(17, 3),
        hooks: Dict[str, List[Callable]] = None,
        fix_bad_data: bool = False,
        **kwargs
    ):
        """
        Args:
            root: root path of dataset.
                If root is a str or Path, it will be regarded as the root path of dataset,
                complete image path is root + image_dirname, complete label path is root + label_dirname;
                if root is a list, the complete image path is root[i], the complete label path is
                root[i].replace(image_dirname, label_dirname)
            with_image: whether the dataset contains images.
            with_label: whether the dataset contains labels.
            image_dirname: the name of image directory.
            label_dirname: the name of label directory.
            task: the task of dataset, chick the value in ['det', 'obb', 'cls', 'seg']
            read_image: whether to read image when get data item.
            transformers: the transformers of dataset.
            subject_to: If the number of images and labels in the dataset do not match, what will prevail
                for the data items of the dataset. Choose in ['image', 'label'].
        """
        super().__init__()
        self._roots_map = {}
        self._image_map = {}
        self._label_map = {}
        self._place_map = {}
        self._with_image = with_image
        self._with_label = with_label
        self._read_image = read_image
        self.image_dirname = image_dirname
        self.label_dirname = label_dirname
        self.task = task
        self.transformers = transformers
        self.subject_to = subject_to
        self.kpt_shape = kpt_shape
        self._index = 0
        self._begin = 0
        self.hooks = hooks if hooks else {}
        self.fix_bad_data = fix_bad_data

        self.image_path_sep = f"{os.sep}{self.image_dirname}{os.sep}"
        self.label_path_sep = f"{os.sep}{self.label_dirname}{os.sep}"

        self._parse_root(root, with_image, image_dirname)
        self._parse_image_label()

        if categories is None:
            with DoNotReadImage(self):
                self.auto_collect_categories()
        else:
            is_name2id = all(isinstance(k, str) for k in categories.keys())
            self._cate_id2name = {v: k for k, v in categories.items()} if is_name2id else categories
            self._cate_name2id = categories if is_name2id else {v: k for k, v in categories.items()}

        with DoNotReadImage(self):
            self.sample_info = self.collect_sample_info() if self._with_label else None

    @property
    def with_image(self):
        return self._with_image

    @property
    def with_label(self):
        return self._with_label

    @property
    def images(self):
        """
        All data absolute paths.
        Returns: list of str
        """
        return [os.path.join(self._roots_map[self._place_map[idx]], name)
                for idx, name in self._image_map.items()]

    @property
    def image_indexes(self):
        return list(self._image_map.keys())

    @property
    def is_read_image(self):
        return self._read_image

    @property
    def directories(self):
        return list(self._roots_map.values())

    @property
    def category_size(self):
        return len(self._cate_id2name)

    def categories(self, key=None):
        if isinstance(key, str):
            return self._cate_name2id.get(key, None)
        elif isinstance(key, int):
            return self._cate_id2name.get(key, None)
        elif key is None:
            return self._cate_id2name
        else:
            raise ValueError("Not expected type of key: {}".format(type(key)))

    def img2label_path(self, image_path, label_postfix=".txt"):
        raise NotImplementedError

    def _parse_root(self, root, with_image, image_dirname):
        if with_image:
            if isinstance(root, list):
                self._roots_map = {idx: r for idx, r in enumerate(root) if self.image_path_sep in r}
            elif isinstance(root, (str, Path)):
                if self.image_path_sep in str(root):
                    self._roots_map = {0: Path(root)}
                else:
                    self._roots_map = {0: Path(root) / image_dirname}
            elif root is None:
                self._with_image = False
            else:
                raise ValueError("Not expected type of root: {}".format(type(root)))
        else:
            if isinstance(root, (str, Path)):
                os.makedirs(root, exist_ok=True)
                self._roots_map[0] = root
            elif root is None:
                pass
            else:
                raise ValueError("When 'with_image=False', the root must be a path or None; "
                                 "If root is a path and doesn't exist, it will be created automatically, "
                                 "if root is None, an empty dataset object is declared.")

    def _parse_image_label(self):
        image_id = 0
        for idx, root in self._roots_map.items():
            if not self.with_image:
                return
            for image_name in os.listdir(root):
                if not image_name.endswith(tuple(IMG_FORMATS)):
                    continue
                self._image_map[image_id] = image_name
                self._place_map[image_id] = idx
                if self.with_label and self.task != "cls":
                    image_path = os.path.join(root, image_name)
                    label_path = self.img2label_path(image_path)
                    self._label_map[image_id] = os.path.basename(label_path) if os.path.exists(label_path) else None
                elif self.with_label and self.task == "cls":
                    category_name = os.path.basename(root)
                    self._label_map[image_id] = category_name
                image_id += 1

    def __iter__(self) -> Iterator[T_co]:
        self._index = self._begin
        return self

    def __next__(self):
        if self._index < len(self):
            ret = self[self._index]
            self._index += 1
            return ret
        else:
            self._index = self._begin
            raise StopIteration(f"Iterator out of range, stop.")

    def __len__(self):
        return len(self._image_map)

    def __getitem__(self, index):
        """
        Enhanced index access, supported in the following forms:
        - dataset[5]        -> Single sample
        - dataset[1:10:2]   -> Slicing generates a new data set
        - dataset[[1,3,5]]  -> The list index generates a new data set

        """
        if isinstance(index, slice):
            return self.subset(_parse_slice(index, self))

        elif isinstance(index, (list, tuple, np.ndarray)):
            return self.subset(_validate_indices(index, self))

        try:
            return self._get_data_item(index) if self._read_image else self._get_single_item(index)
        except KeyError:
            raise IndexError(f"index {index} out of dataset range [0, {len(self) - 1}]")

    def __setitem__(self, index: int, data: Union[str, tuple]):
        """
        Support two assignment modes:
        1. Only update the image path (when with_label=False)
           example: dataset[0] = "/new/path/img1.jpg"
        2. Update the image path and label simultaneously (when with_label=True)
           example: dataset[1] = ("/new/path/img2.jpg", "/new/path/lab2.txt")
        """
        if index < -len(self) or index >= len(self):
            raise IndexError(f"Index {index} out of dataset range [0, {len(self) - 1}].")
        if index < 0:
            index += len(self)

        if self.with_label:
            if not isinstance(data, tuple) or len(data) != 2:
                raise ValueError("The (path, label) tuple is required.")
            new_image, new_label = data
        else:
            if not isinstance(data, str):
                raise ValueError("need to provide an image path string.")
            new_image = data
            new_label = None

        image_dir = os.path.dirname(new_image)
        image_name = os.path.basename(new_image)

        root_idx = None
        for idx, r_path in self._roots_map.items():
            if os.path.abspath(r_path) == os.path.abspath(image_dir):
                root_idx = idx
                break
        if root_idx is None:
            raise ValueError(f"Image directory {image_dir} not found in dataset roots.")

        self._image_map[index] = image_name
        self._place_map[index] = root_idx

        if self.with_label and self.task in ['det', 'obb', 'seg']:
            if new_label is not None:
                if os.path.exists(new_label):
                    self._label_map[index] = os.path.basename(new_label)
                else:
                    self._label_map[index] = None
            else:
                label_path = self.img2label_path(new_image)
                if os.path.exists(label_path):
                    self._label_map[index] = os.path.basename(label_path)
                else:
                    self._label_map[index] = None
        elif self.with_label and self.task in ['cls']:
            if new_label is not None:
                self._label_map[index] = new_label if isinstance(new_label, str) else str(new_label)

    def _get_single_item(self, index):
        if self._begin <= index < len(self):
            root_idx = self._place_map[index]
            root_path = self._roots_map[root_idx]
            image_name = self._image_map[index]
            image_path = os.path.join(root_path, image_name)

            if self.with_label:
                return image_path, self.img2label_path(image_path)
            return image_path, None

        raise IndexError(f"Index {index} is out of the valid range")

    def _get_data_item(self, index):
        image_path, label_path = self._get_single_item(index)
        image = imread(image_path)
        label = None

        if self.with_label and label_path and os.path.exists(label_path):
            label = self._parse_label_file(label_path, fix_data=self.fix_bad_data)

        if self.transformers:
            transformed = self.transformers(image=image, label=label)
            image = transformed['image']
            label = transformed['label']

        return image_path, label_path, image, label

    def _parse_label_file(self, *args, **kwargs):
        """
        Parse format annotation file for different tasks.
        """
        raise NotImplementedError

    def subset(self, indices: Union[list]):
        subset = eval(self.__class__.__name__)(
            root=None,
            with_image=self.with_image,
            with_label=self.with_label,
            image_dirname=self.image_dirname,
            label_dirname=self.label_dirname,
            categories=self._cate_id2name,
            task=self.task,
            read_image=self._read_image,
            transformers=self.transformers,
            subject_to=self.subject_to,
            hooks=self.hooks,
            kpt_shape=self.kpt_shape,
        )

        used_roots = set(self._place_map[idx] for idx in indices)
        subset._roots_map = {idx: self._roots_map[idx] for idx in used_roots}

        subset._image_map = {}
        subset._label_map = {}
        subset._place_map = {}
        for new_idx, old_idx in enumerate(indices):
            subset._image_map[new_idx] = self._image_map[old_idx]
            subset._place_map[new_idx] = self._place_map[old_idx]
            if self.with_label:
                subset._label_map[new_idx] = self._label_map.get(old_idx)

        return subset

    def append(self, image_path, label=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found.")

        image_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)

        if self.image_dirname not in image_dir.split(os.sep):
            raise ValueError(f"Image directory must contain {self.image_dirname}.")

        root_idx = None
        for idx, r_path in self._roots_map.items():
            if r_path == image_dir:
                root_idx = idx
                break
        if root_idx is None:
            root_idx = max(self._roots_map.keys(), default=-1) + 1
            self._roots_map[root_idx] = image_dir

        new_idx = len(self)
        self._image_map[new_idx] = image_name
        self._place_map[new_idx] = root_idx

        if self.with_label:
            if label and os.path.exists(label):
                self._label_map[new_idx] = os.path.basename(label)
            else:
                label_path = self.img2label_path(image_path)
                self._label_map[new_idx] = os.path.basename(label_path) if os.path.exists(label_path) else None

    def split(self, ratio: Union[float, List[float]] = None, subset_name=None, seed: int = None, grouped: bool = True, specified=None):
        return split(self, ratio=ratio, subset_name=subset_name, seed=seed, grouped=grouped, specified=specified)

    def get_root_groups(self):
        """Gets an index of samples grouped by root directory"""
        groups = defaultdict(list)
        for idx in self._image_map:
            root_id = self._place_map[idx]
            groups[root_id].append(idx)
        return groups

    def run_hooks(self, name: str, *args, **kwargs):
        for hook in self.hooks.get(name, []):
            hook(*args, **kwargs)

    def auto_collect_categories(self):
        raise NotImplementedError

    def collect_sample_info(self):
        raise NotImplementedError


class YOLODataset(SeparateDataset):

    def __init__(self, *args, image_dirname="images", label_dirname="labels", **kwargs):
        """
        Args:
            root: root path of dataset.
                If root is a str or Path, it will be regarded as the root path of dataset,
                complete image path is root + image_dirname, complete label path is root + label_dirname;
                if root is a list, the complete image path is root[i], the complete label path is
                root[i].replace(image_dirname, label_dirname)
            with_image: whether the dataset contains images.
            with_label: whether the dataset contains labels.
            image_dirname: the name of image directory.
            label_dirname: the name of label directory.
            task: the task of dataset, chick the value in ['det', 'obb', 'cls', 'seg']
            read_image: whether to read image when get data item.
            transformers: the transformers of dataset.
            subject_to: If the number of images and labels in the dataset do not match, what will prevail
                for the data items of the dataset. Choose in ['image', 'label'].
        """
        super().__init__(*args, image_dirname=image_dirname, label_dirname=label_dirname, **kwargs)

    def img2label_path(self, image_path, label_postfix=".txt"):
        return self.label_path_sep.join(image_path.rsplit(self.image_path_sep, 1)).rsplit(".", 1)[0] + label_postfix

    def _parse_label_file(self, label_path, encoding='utf-8', fix_data=False):
        """
        Parse YOLO format annotation file for different tasks.

        Args:
            label_path: Path to the label file
            encoding: File encoding (default: utf-8)

        Returns:
            List of annotations in the format:
            - det: [(class_id, x_center, y_center, width, height), ...]
            - obb: [(class_id, x1, y1, x2, y2, x3, y3, x4, y4), ...]
            - seg: [(class_id, [x1, y1, x2, y2, ...]), ...]
            - pose: [(class_id, x_center, y_center, width, height, [(px1, py1, v1?), (px2, py2, v2?), ...]), ...]

        Raises:
            ValueError: If label format doesn't match task requirements
        """

        annotations = []

        try:
            with open(label_path, 'r', encoding=encoding) as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except UnicodeDecodeError:
            # Fallback to other common encodings if utf-8 fails
            try:
                with open(label_path, 'r', encoding='latin-1') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
            except Exception as e:
                raise ValueError(f"Failed to read label file {label_path}: {str(e)}")

        for line_num, line in enumerate(lines, 1):
            parts = line.split()
            if not parts:
                continue

            try:
                class_id = int(parts[0])
                values = list(map(float, parts[1:]))

                if self.task == 'det':
                    # Expected format: class_id x_center y_center width height
                    if len(values) != 4:
                        raise ValueError(f"Detection label requires 4 values, got {len(values)}")
                    validate_normalized_coords(values, line_num, fix_data)
                    x, y, w, h = values
                    la = (class_id, x, y, w, h)

                elif self.task == 'obb':
                    # Expected format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    if len(values) != 8:
                        raise ValueError(f"OBB label requires 8 values, got {len(values)}")
                    validate_normalized_coords(values, line_num, fix_data)
                    la = (class_id, *values)

                elif self.task == 'seg':
                    # Expected format: class_id x1 y1 x2 y2 ... (at least 3 points)
                    if len(values) < 6 or len(values) % 2 != 0:
                        raise ValueError(f"Segmentation label requires even number of values (>=6), got {len(values)}")
                    validate_normalized_coords(values, line_num, fix_data)
                    points = [(values[i], values[i+1]) for i in range(0, len(values), 2)]
                    la = (class_id, points)

                elif self.task == 'pose':
                    # Expected format:
                    # class_id x_center y_center width height kp1_x kp1_y <p1-visibility> kp2_x kp2_y <p2-visibility>...
                    if len(values) < 4:
                        raise ValueError(f"Pose label requires at least 4 values, got {len(values)}")
                    bbox = values[:4]
                    validate_normalized_coords(bbox, line_num, fix_data)
                    kps = values[4:]

                    # Auto-detect pose format (2D or 3D)
                    keypoints = [
                        (kps[i], kps[i + 1], int(kps[i + 2]) if self.kpt_shape[-1] == 3 else 1)
                        for i in range(0, len(kps), self.kpt_shape[-1])
                    ]
                    la = (class_id, *bbox, keypoints)

                else:
                    raise ValueError(f"Unsupported task type: {self.task}")

                annotations.append(la)

            except (ValueError, IndexError) as e:
                raise ValueError(f"Invalid label format in file: '{label_path}'. Error: {str(e)}")

        return annotations

    def auto_collect_categories(self):
        if not self.with_label:
            return
        categories = set()
        for _, label_path in self:
            if label_path and os.path.exists(label_path):
                label = self._parse_label_file(label_path, fix_data=self.fix_bad_data)
                for la in label:
                    categories.add(la[0])
        self._cate_id2name = {cid: str(cid) for cid in categories}
        self._cate_name2id = {v: k for k, v in self._cate_id2name.items()}

    def collect_sample_info(self):
        sample_info = dict()
        for _, label_path in self:
            if label_path and os.path.exists(label_path):
                label = self._parse_label_file(label_path, fix_data=self.fix_bad_data)
                for la in label:
                    sample_info[
                        self._cate_id2name.get(la[0], la[0])
                    ] = sample_info.get(
                        self._cate_id2name.get(la[0], la[0]), 0
                    ) + 1
        return sample_info


def split(
    dataset,
    ratio: Union[float, List[float]] = None,
    subset_name=None,
    seed: int = None,
    grouped: bool = True,
    specified=None
):
    if ratio is None:
        ratio = [0.6, 0.2, 0.2]
    elif isinstance(ratio, float):
        ratio = [ratio, 1 - ratio]
    if subset_name is None:
        subset_name = ['train', 'val', 'test']
    if abs(sum(ratio) - 1.0) > 1.0e-6:
        raise ValueError("Sum of ratio must be 1, current sum ratio: ", sum(ratio), abs(1 - sum(ratio)))
    if len(ratio) > len(subset_name):
        raise ValueError("Length of ratio must be less than or equal to length of subset_name.")

    if specified is not None and len(specified) > len(subset_name):
        raise ValueError("Length of specified must be less than or equal to length of subset_name.")

    if seed is not None:
        random.seed(seed)

    shuffled_list = []
    if grouped:
        for gid, group in dataset.get_root_groups().items():
            random.shuffle(group)
            shuffled_list.append(group)
    else:
        shuffled = dataset.image_indexes
        random.shuffle(shuffled)
        shuffled_list = [shuffled]

    subsets = {}
    for shuffled in shuffled_list:
        total = len(shuffled)
        lengths = []
        start_idx = 0
        for i in range(len(ratio)):
            if i < len(ratio) - 1:
                length = int(ratio[i] * total)
            else:
                length = total - sum(lengths)
            lengths.append(length)
            end_idx = start_idx + length
            subsets.setdefault(subset_name[i], []).extend(shuffled[start_idx:end_idx])
            start_idx = end_idx
    for name, subset in subsets.items():
        subsets[name] = sorted(subset)

    return subsets


def validate_normalized_coords(coords, l_n, fix_data=False):
    """Validate coordinates are normalized (0-1)"""
    for i, val in enumerate(coords):
        if not (0 <= val <= 1):
            if fix_data:
                coords[i] = max(0.0, min(1.0, val))
            else:
                raise ValueError(
                    f"Line {l_n}: Coordinate {i} out of range [0,1]: {val}"
                )


def dump_yolo_dataset(
    dataset: YOLODataset,
    destination: str,
    image_file_op: Union[str, Callable] = "copy",
    label_file_op: Union[str, Callable] = "copy",
    image_dirname: str = "images",
    label_dirname: str = "labels",
    sub_dirname: str = "",
    tqdm_enable: bool = True,
):
    if not dataset:
        raise ValueError("Dataset is empty.")
    op = {
        "copy": shutil.copy,
        "move": shutil.move
    }
    if isinstance(image_file_op, str):
        img_op = op[image_file_op]
    elif isinstance(image_file_op, Callable):
        img_op = image_file_op
    else:
        raise ValueError(f"Unsupported image file operation: {image_file_op}")
    if isinstance(label_file_op, str):
        lab_op = op[label_file_op]
    elif isinstance(label_file_op, Callable):
        lab_op = label_file_op
    else:
        raise ValueError(f"Unsupported label file operation: {label_file_op}")

    try:
        iterator = tqdm(dataset, desc=f"Dumping dataset") if tqdm_enable else dataset
        for idx, (old_image_path, old_label_path) in enumerate(iterator):
            image_name = old_image_path.rsplit(dataset.image_path_sep, 1)[1]
            if sub_dirname == "":
                path_piece = [destination, image_dirname, image_name]
            else:
                path_piece = [destination, image_dirname, sub_dirname, image_name]
            new_image_path = os.path.join(*path_piece)
            new_label_path = img2label_path(new_image_path, image_dirname, label_dirname)
            if os.path.exists(new_image_path) or os.path.exists(new_label_path):
                print("Exists:", new_image_path, new_label_path)
                continue
            new_image_dir = os.path.dirname(new_image_path)
            os.makedirs(new_image_dir, exist_ok=True)
            new_label_dir = os.path.dirname(new_label_path)
            os.makedirs(new_label_dir, exist_ok=True)
            img_op(old_image_path, new_image_path)
            lab_op(old_label_path, new_label_path)
    except Exception as e:
        raise e


class VOCDataset(SeparateDataset):

    def __init__(self, *args, label_dirname="Annotations", **kwargs):
        """
        Args:
            root: root path of dataset.
                If root is a str or Path, it will be regarded as the root path of dataset,
                complete image path is root + image_dirname, complete label path is root + label_dirname;
                if root is a list, the complete image path is root[i], the complete label path is
                root[i].replace(image_dirname, label_dirname)
            with_image: whether the dataset contains images.
            with_label: whether the dataset contains labels.
            image_dirname: the name of image directory.
            label_dirname: the name of label directory.
            task: the task of dataset, chick the value in ['det', 'obb', 'cls', 'seg']
            read_image: whether to read image when get data item.
            transformers: the transformers of dataset.
            subject_to: If the number of images and labels in the dataset do not match, what will prevail
                for the data items of the dataset. Choose in ['image', 'label'].
        """
        super().__init__(*args, label_dirname=label_dirname, **kwargs)

    def img2label_path(self, image_path, label_postfix=".xml"):
        return self.label_path_sep.join(image_path.rsplit(self.image_path_sep, 1)).rsplit(".", 1)[0] + label_postfix

    def _parse_label_file(self, label_path):
        """
        Parse YOLO format annotation file for different tasks.

        Args:
            label_path: Path to the label file

        Returns:

        Raises:
            ValueError: If label format doesn't match task requirements
        """

        data = XMLParser.load(label_path)
        return data

    def auto_collect_categories(self):
        if not self.with_label:
            return
        categories = set()
        for _, label_path in self:
            if label_path and os.path.exists(label_path):
                label = self._parse_label_file(label_path)["annotation"]
                if "object" in label.keys():
                    if isinstance(label["object"], list):
                        for obj in label["object"]:
                            categories.add(obj["name"])
                    else:
                        if "name" in label["object"].keys():
                            categories.add(label["object"]["name"])
                else:
                    print("object is null")
        self._cate_id2name = {cid: name for cid, name in enumerate(categories)}
        self._cate_name2id = {v: k for k, v in self._cate_id2name.items()}

    def collect_sample_info(self):
        sample_info = dict()
        for _, label_path in self:
            if label_path and os.path.exists(label_path):
                label = self._parse_label_file(label_path)["annotation"]
                if "object" in label.keys():
                    if isinstance(label["object"], list):
                        for obj in label["object"]:
                            sample_info[obj["name"]] = sample_info.get(obj["name"], 0) + 1
                    else:
                        if "name" in label["object"].keys():
                            sample_info[obj["name"]] = sample_info.get(obj["name"], 0) + 1
                else:
                    print("object is null")
        return sample_info
