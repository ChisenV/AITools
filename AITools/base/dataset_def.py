import bisect
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    Dict
)

__all__ = [
    "T_co",
    "T",
    "T_dict",
    "T_tuple",
    "T_stack",
    "Dataset",
    "IterableDataset",
    "StackDataset",
    "ConcatDataset",
    "ChainDataset",
    "Subset"
]

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
T_dict = Dict[str, T_co]
T_tuple = Tuple[T_co, ...]
T_stack = TypeVar('T_stack', T_tuple, T_dict)


class _Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    # def __getitems__(self, indices: List) -> List[T_co]:
    # Not implemented to prevent false-positives in fetcher check in
    # torch.utils.data._utils.fetch._MapDatasetFetcher

    def __add__(self, other: '_Dataset[T_co]') -> '_ConcatDataset[T_co]':
        return _ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in sampler.py


class _IterableDataset(_Dataset[T_co]):
    r"""An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overwrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.
    """

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __add__(self, other: _Dataset[T_co]):
        return _ChainDataset([self, other])

    # No `def __len__(self)` default? Subclasses raise `TypeError` when needed.
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]


class _StackDataset(_Dataset[T_stack]):
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {'image': images[0], 'text': texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """
    datasets: Union[tuple, dict]

    def __init__(self, *args: _Dataset[T_co], **kwargs: _Dataset[T_co]) -> None:
        if args:
            if kwargs:
                raise ValueError("Supported either ``tuple``- (via ``args``) or"
                                 "``dict``- (via ``kwargs``) like input/output, but both types are given.")
            self._length = len(args[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in args):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])  # type: ignore[arg-type]
            if any(self._length != len(dataset) for dataset in tmp):  # type: ignore[arg-type]
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs
        else:
            raise ValueError("At least one dataset should be passed")

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return self._length


class _ConcatDataset(_Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[_Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[_Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, _IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        # warnings.warn("cummulative_sizes attribute is renamed to "
        #               "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class _ChainDataset(_IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    def __init__(self, datasets: Iterable[_Dataset]) -> None:
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, _IterableDataset), "ChainDataset only supports IterableDataset"
            yield from d

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, _IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)  # type: ignore[arg-type]
        return total


class _Subset(_Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: _Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: _Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]) -> List[T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)


try:
    from torch.utils.data import (
        Dataset,
        IterableDataset,
        StackDataset,
        ConcatDataset,
        ChainDataset,
        Subset
    )
except ImportError:
    Dataset = _Dataset
    IterableDataset = _IterableDataset
    StackDataset = _StackDataset
    ConcatDataset = _ConcatDataset
    ChainDataset = _ChainDataset
    Subset = _Subset
