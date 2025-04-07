import concurrent
import copy
import json
import math
import os
import random
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Union
)

import numpy as np

__all__ = [
    "OCRDatasetV2",
    "OCRCLSDatasetV2",
    "OCRRECDatasetV2"
]

from AITools.base.dataset_def import IterableDataset, T_co
from AITools.base.vision_def import IMG_FORMATS
from AITools.core.manager import ComponentManager
from AITools.comp.functions import parse_ppocr_label

DATASETS = ComponentManager("datasets")


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
        self._root_dirs = set()  # New: Quick directory existence check
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

        if abs_image_root not in self._root_dirs:
            if root_idx is None:
                root_idx = len(self._roots_map)
                self._roots_map[root_idx] = abs_image_root
            self._root_dirs.add(abs_image_root)
            self._dir_basename_map[abs_image_root] = set()
        else:
            root_idx = [k for k, v in self._roots_map.items() if v == abs_image_root][0]

        label_file = self._lab_files.get(root_idx, None)
        data = {} if label_file is None else self._parse_file(label_file, encoding=label_file_encoding)
        if self.subject_to == "label" and self.with_label:
            for i, (im, la) in enumerate(data.items()):
                idx = offset + i
                img_basename = os.path.basename(im)
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
                data[image_name] = self._parse_label(a)
        return data

    def fmt_label_dumps(self, label):
        return (str(label).replace("'", '"')
                .replace('"difficult": 0', '"difficult": false')
                .replace('"difficult": 1', '"difficult": true'))

    def fmt_label_loads(self, label: str):
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
            return self.subset(self._parse_slice(index))

        elif isinstance(index, (list, tuple, np.ndarray)):
            return self.subset(self._validate_indices(index))

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

    def _parse_slice(self, s: slice) -> list:
        """Converts slice objects to a list of valid indexes"""
        start, stop, step = s.indices(len(self))
        return list(range(start, stop, step))

    def _validate_indices(self, indices) -> list:
        """Validate and filter the index list"""
        max_idx = len(self) - 1
        return [idx for idx in indices if 0 <= idx <= max_idx]

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

    def _update_directory_mappings(self):
        """更新子数据集的目录映射关系"""
        # rebuild roots_map
        active_roots = set(self._place_map.values())
        self._roots_map = {rid: path for rid, path in self._roots_map.items() if rid in active_roots}

        # rebuild root_dirs
        self._root_dirs = {path for path in self._roots_map.values()}

        # rebuild dir_basename_map
        self._dir_basename_map = defaultdict(set)
        for idx in self._image_map:
            root_id = self._place_map[idx]
            root_path = self._roots_map[root_id]
            self._dir_basename_map[root_path].add(self._image_map[idx])

    def __setitem__(self, index: int, data: Union[str, tuple]):
        """
        支持两种赋值模式：
        1. 仅更新图像路径 (当 with_label=False 时)
           示例：dataset[0] = "/new/path/img1.jpg"
        2. 同时更新图像路径和标签 (当 with_label=True 时)
           示例：dataset[1] = ("/new/path/img2.jpg", {"text": "Hello"})
        """
        # ==================== 参数校验 ====================
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

        # ==================== 路径处理 ====================
        # 标准化新路径
        abs_new_path = os.path.abspath(os.path.normpath(new_image_path))
        new_dir = os.path.dirname(abs_new_path)
        new_basename = os.path.basename(abs_new_path)

        # 获取原数据信息
        old_basename = self._image_map[index]
        old_root_id = self._place_map[index]
        old_dir = self._roots_map[old_root_id]

        # ==================== 路径冲突检查 ====================
        # 检查新文件名是否在目标目录已存在
        if new_dir == old_dir:
            # 同目录更新文件名
            if new_basename != old_basename and new_basename in self._dir_basename_map[old_dir]:
                raise FileExistsError(f"File {new_basename} already exists in directory {old_dir}.")
            new_root_id = old_root_id
        else:
            # 跨目录更新需要检查新目录
            if new_dir in self._root_dirs and new_basename in self._dir_basename_map[new_dir]:
                raise FileExistsError(f"File {new_basename} already exists in directory {new_dir}")

            # 自动注册新目录
            new_root_id = len(self._roots_map)
            self._roots_map[new_root_id] = new_dir
            self._root_dirs.add(new_dir)
            self._dir_basename_map[new_dir] = set()

        # ==================== 更新数据结构 ====================
        # 1. 清理旧记录
        self._dir_basename_map[old_dir].discard(old_basename)

        # 2. 更新核心映射
        self._image_map[index] = new_basename
        self._place_map[index] = new_root_id
        self._dir_basename_map[new_dir].add(new_basename)

        # 3. 更新标签 (如果启用)
        if self.with_label:
            self._label_map[index] = new_label

        # 4. 自动清理空目录
        if not self._dir_basename_map[old_dir]:
            del self._roots_map[old_root_id]
            self._root_dirs.discard(old_dir)
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
                merged_dataset._root_dirs.add(root_path)
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
            #
            other_item = other[other_idx]
            abs_path = os.path.abspath(
                other_item[0] if isinstance(other_item, tuple) else other_item
            )
            dir_path = os.path.dirname(abs_path)
            base_name = os.path.basename(abs_path)

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

    def append(self, image_path, label=None):
        abs_image_path = os.path.abspath(os.path.normpath(image_path))
        if not os.path.exists(abs_image_path) or not abs_image_path.endswith(tuple(IMG_FORMATS)):
            raise ValueError(f"File '{abs_image_path}' does not exist or is not an image file.")
        dirname = os.path.dirname(abs_image_path)
        basename = os.path.basename(abs_image_path)
        if dirname in self._root_dirs and basename in self._dir_basename_map[dirname]:
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
        if dirname in self._root_dirs:
            root_idx = [k for k, v in self._roots_map.items() if v == dirname][0]
        else:
            root_idx = len(self._roots_map)
            self._roots_map[root_idx] = dirname
            self._root_dirs.add(dirname)
            self._dir_basename_map[dirname] = set()
        self._image_map[idx] = basename
        self._place_map[idx] = root_idx
        self._dir_basename_map[dirname].add(basename)

    def split(self, ratio: Union[float, List[float]] = None, subset_name=None, seed: int = None, specified=None):
        if ratio is None:
            ratio = [0.6, 0.2, 0.2]
        elif isinstance(ratio, float):
            ratio = [ratio, 1 - ratio]
        if subset_name is None:
            subset_name = ['train', 'val', 'test']
        if sum(ratio) != 1:
            raise ValueError("Sum of ratio must be 1.")
        if len(ratio) > len(subset_name):
            raise ValueError("Length of ratio must be less than or equal to length of subset_name.")

        if specified is not None and len(specified) > len(subset_name):
            raise ValueError("Length of specified must be less than or equal to length of subset_name.")

        if seed is not None:
            random.seed(seed)

        shuffled = list(self._image_map.keys())
        random.shuffle(shuffled)
        total = len(shuffled)

        lengths = []
        for i in range(len(ratio)):
            if i < len(ratio) - 1:
                length = int(ratio[i] * total)
                lengths.append(length)
            else:
                lengths.append(total - sum(lengths))

        subsets = {}
        start_idx = 0
        for i, length in enumerate(lengths):
            end_idx = start_idx + length
            subsets[subset_name[i]] = sorted(shuffled[start_idx:end_idx])
            start_idx = end_idx

        return subsets

    def wash(self, image_list: List[Union[str, int]], mode='drop'):
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
        self._root_dirs = set(new_roots_map.values())
        self._dir_basename_map = defaultdict(set)
        for new_index in new_image_map.keys():
            root_dir = new_roots_map[new_place_map[new_index]]
            img_name = new_image_map[new_index]
            self._dir_basename_map[root_dir].add(img_name)

    def copy(self) -> Any:
        """Creates a deep-copy copy of the current dataset"""
        new = OCRDatasetV2.__new__(OCRDatasetV2)  # Avoid calling __init__

        new._image_map = copy.deepcopy(self._image_map)
        new._label_map = copy.deepcopy(self._label_map)
        new._place_map = copy.deepcopy(self._place_map)
        new._roots_map = copy.deepcopy(self._roots_map)
        new._lab_files = copy.deepcopy(self._lab_files)
        new._root_dirs = copy.deepcopy(self._root_dirs)
        new._dir_basename_map = copy.deepcopy(self._dir_basename_map)
        new._index = self._index
        new._begin = self._begin

        new._with_image = self._with_image
        new._with_label = self._with_label
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
        if condition is not None and not callable(condition):
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

    def fmt_label_dumps(self, label: str):
        return label.strip("\n").strip("\r")

    def fmt_label_loads(self, label: str):
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
