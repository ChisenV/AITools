import tensorrt as trt

from AITools.components.backend import DeployModel


class TensorRTModel(DeployModel):
    """TensorRT model inference implementation"""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.engine = None
        self.context = None
        self.bindings = []

    def load_model(self) -> None:
        """Load TensorRT engine and create execution context"""
        self.logger.info(f"Loading TensorRT engine from {self.model_path}")

        # Initialize TensorRT runtime
        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)

        # Deserialize engine
        with open(self.model_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # Create execution context
        self.context = self.engine.create_execution_context()

        # Get input/output names from engine
        self.input_names = [
            self.engine.get_binding_name(i)
            for i in range(self.engine.num_bindings)
            if self.engine.binding_is_input(i)
        ]
        self.output_names = [
            self.engine.get_binding_name(i)
            for i in range(self.engine.num_bindings)
            if not self.engine.binding_is_input(i)
        ]

        super().load_model()

    def destroy(self) -> None:
        """Release TensorRT resources"""
        self.logger.info("Releasing TensorRT resources")
        if self.context:
            del self.context
        if self.engine:
            del self.engine
        super().destroy()
