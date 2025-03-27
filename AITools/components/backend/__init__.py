import abc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class DeployModel(abc.ABC):
    """Base class for AI model inference with standardized processing pipeline"""

    def __init__(
            self,
            model_path: str,
            input_names: Optional[List[str]] = None,
            output_names: Optional[List[str]] = None,
            log_level: int = logging.INFO,
    ):
        """
        Initialize base AI model

        :param model_path: Path to the model file
        :param input_names: List of input tensor names (optional)
        :param output_names: List of output tensor names (optional)
        :param log_level: Logging level (default: INFO)
        """
        self.model_path = Path(model_path)
        self.input_names = input_names or []
        self.output_names = output_names or []
        self.logger = self._setup_logger(log_level)
        self._initialized = False

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def _setup_logger(self, level: int) -> logging.Logger:
        """Configure logger instance"""
        logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        return logger

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load model and initialize inference engine"""
        self._initialized = True

    @abc.abstractmethod
    def destroy(self) -> None:
        """Release resources and clean up"""
        self._initialized = False

    @abc.abstractmethod
    def preprocess(self, input_data: Any) -> Dict[str, np.ndarray]:
        """
        Preprocess input data to model-ready format

        :param input_data: Raw input data (e.g., image path, numpy array)
        :return: Dictionary of preprocessed inputs {input_name: tensor}
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_model() first")

    @abc.abstractmethod
    def inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute model inference

        :param inputs: Preprocessed input tensors
        :return: Raw model outputs {output_name: tensor}
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_model() first")

    @abc.abstractmethod
    def postprocess(
            self, outputs: Dict[str, np.ndarray], **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        """
        Convert raw model outputs to final results

        :param outputs: Raw model outputs
        :param kwargs: Additional postprocessing parameters
        :return: Tuple of (main results, optional secondary outputs)
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call load_model() first")

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


try:
    import tensorrt
    from .tensorrt import *
except ImportError:
    pass


