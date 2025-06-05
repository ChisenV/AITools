已有：
```python
class BaseModelHandler(ABC):

    _SUPPORTED_EXTENSIONS = []

    def __init__(
            self,
            model_path: Union[str, Path],
            config: Union[Config, Dict[str, Any]],
            **kwargs
    ):
        self.model_path = Path(model_path)
        self.config = config
        self.config.update(kwargs)
        self._initialized = False

    @abstractmethod
    def load(self, *args, **kwargs):
        """Load the model and initialize compute resources (such as GPU bindings)"""
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Perform inference, return original output (no post-processing)"""
        pass

    @abstractmethod
    def destroy(self, *args, **kwargs):
        """Free up resources occupied by the model (e.g. video memory, thread pool)"""
        pass

class _Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    """
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __add__(self, other: '_Dataset[T_co]') -> '_ConcatDataset[T_co]':
        return _ConcatDataset([self, other])
    
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")
```
如何设计一个Evaluator用于评估模型推理的结果，给出对应的指标数据，主要是分类任务、检测任务、语义分割任务、实例分割任务、OCR识别任务，先给出详细设计与框架，不用完整的实现
