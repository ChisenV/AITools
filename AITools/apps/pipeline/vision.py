# TODO demo
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional
import asyncio
import time

from AITools import Config


class DataBatch:
    """增强型数据批处理容器"""

    def __init__(self,
                 raw_data: Any,
                 metadata: Optional[Dict] = None,
                 inference_result: Optional[Any] = None,
                 evaluation_result: Optional[Dict[str, float]] = None):
        self.raw_data = raw_data  # 原始输入数据
        self.metadata = metadata or {}  # 数据元信息
        self.inference_result = inference_result  # 模型推理结果
        self.evaluation_result = evaluation_result  # 评估指标
        self.timestamp = time.monotonic()  # 处理时间戳


class AsyncDataSource(ABC):
    """异步数据源抽象基类"""

    @abstractmethod
    async def data_stream(self) -> AsyncGenerator[Any, None]:
        """必须实现为异步数据生成器"""
        yield


class ModelInference(ABC):
    """增强型模型推理接口"""

    @abstractmethod
    async def initialize(self):
        """异步初始化模型"""
        pass

    @abstractmethod
    async def inference(self, batch: DataBatch) -> DataBatch:
        """执行异步推理"""
        pass

    async def shutdown(self):
        """资源清理（可选实现）"""
        pass


class Evaluator(ABC):
    """增强型评估器接口"""

    def __init__(self):
        self.global_metrics = {}

    @abstractmethod
    async def evaluate(self, batch: DataBatch) -> DataBatch:
        """执行异步评估"""
        pass

    @abstractmethod
    async def summarize(self) -> Dict[str, float]:
        """返回全局评估指标"""
        pass

    async def reset(self):
        """重置评估状态"""
        self.global_metrics.clear()


class ResultSaver(ABC):
    """增强型结果存储器"""

    @abstractmethod
    async def initialize(self):
        """异步初始化存储资源"""
        pass

    @abstractmethod
    async def save(self, batch: DataBatch):
        """异步保存结果"""
        pass

    async def finalize(self):
        """最终清理操作（可选实现）"""
        pass


class Pipeline:
    """内存安全的异步处理流水线"""

    def __init__(
            self,
            config: Config,
            model: ModelInference,
            evaluator: Evaluator,
            saver: ResultSaver,
            max_concurrent: int = 4,
            max_queue_size: int = 100,
            throughput_window: int = 10
    ):
        self.model = model
        self.evaluator = evaluator
        self.saver = saver
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_queue_size = max_queue_size
        self.throughput = 0
        self.processed_count = 0
        self.throughput_window = throughput_window
        self.last_update = time.monotonic()

    async def stream_data(
            self,
            data_source: AsyncDataSource,
            batch_size: int = 32
    ):
        """安全的数据流注入方法"""
        async for data in data_source.data_stream():
            # 动态背压控制
            while self.queue.qsize() >= self.max_queue_size * 0.9:
                await asyncio.sleep(0.1)

            await self.queue.put(DataBatch(raw_data=data))

            # 动态批处理（示例）
            if self.queue.qsize() % batch_size == 0:
                await self.queue.join()

    async def _process_batch(self):
        """增强型批处理协程"""
        async with self.semaphore:
            while True:
                try:
                    batch = await self.queue.get()

                    # 执行推理
                    batch = await self.model.inference(batch)

                    # 执行评估
                    batch = await self.evaluator.evaluate(batch)

                    # 保存结果
                    await self.saver.save(batch)

                    # 更新吞吐量统计
                    self._update_throughput()

                except Exception as e:
                    await self.handle_error(e)
                finally:
                    self.queue.task_done()

    def _update_throughput(self):
        """吞吐量计算"""
        self.processed_count += 1
        now = time.monotonic()
        if now - self.last_update > self.throughput_window:
            self.throughput = self.processed_count / (now - self.last_update)
            self.processed_count = 0
            self.last_update = now

    async def run(self, num_workers: int = 4):
        """增强型流水线运行"""
        await self.model.initialize()
        await self.saver.initialize()

        workers = [asyncio.create_task(self._process_batch())
                   for _ in range(num_workers)]

        # 启动监控任务
        monitor_task = asyncio.create_task(self._monitor_resources())

        try:
            await self.queue.join()
        finally:
            for worker in workers:
                worker.cancel()
            monitor_task.cancel()

            await self.model.shutdown()
            await self.saver.finalize()

    async def _monitor_resources(self):
        """资源监控后台任务"""
        while True:
            if self.queue.qsize() > self.max_queue_size * 0.8:
                await self._adjust_throughput()
            await asyncio.sleep(1)

    async def _adjust_throughput(self):
        """动态吞吐量调节"""
        target_speed = self.throughput * 0.9
        # 实现具体的调节逻辑（如调整worker数量）

    async def get_final_report(self) -> Dict[str, float]:
        """获取最终评估报告"""
        return await self.evaluator.summarize()

    async def emergency_drain(self):
        """内存紧急释放"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def handle_error(self, error: Exception):
        """增强型错误处理"""
        print(f"Error occurred: {str(error)}")
        # 可扩展重试逻辑、错误上报等
