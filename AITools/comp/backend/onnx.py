from typing import Dict

import numpy as np

from . import BACKENDS
from AITools.base.model_def import BaseModelHandler


@BACKENDS.register_component
class ONNXHandler(BaseModelHandler):
    def load(self):
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        if self.config.get("enable_profiling", False):
            sess_options.enable_profiling = True
        # 配置执行提供者（如 CUDA/TensorRT）
        providers = self._select_providers()
        self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        # 提取输入/输出层元信息
        self.model_info["inputs"] = {inp.name: inp.shape for inp in self.session.get_inputs()}
        self.model_info["outputs"] = [out.name for out in self.session.get_outputs()]
        self._initialized = True

    def run(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.session.run(None, input_data)

    def destroy(self):
        del self.session
        self._initialized = False