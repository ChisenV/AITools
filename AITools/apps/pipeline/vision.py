from collections import deque, defaultdict

from AITools import Config, Builder

# class BaseResultSaver(ABC):
#     """结果保存基类"""
#
#     def __init__(self, output_dir: Union[str, Path]):
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)
#
#     @abstractmethod
#     def save_batch(self, batch_data: Dict[str, Any], batch_id: int):
#         """保存批次结果"""
#         pass
#
#     def finalize(self):
#         """完成所有保存操作（如合并临时文件）"""
#         pass
#
#
# class ModelInferWorkflow(BaseProcessor):
#     """增强版工作流基类"""
#
#     def __init__(self, config: Union[Config, Dict[str, Any]], model: BaseModelHandler, dataset: IterableDataset,
#                  preprocessor: BasePreprocessor, postprocessor: BasePostprocessor, saver: BaseResultSaver,
#                  mode: str = 'inference', evaluator: Optional[BaseEvaluator] = None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.config = config
#         self.model = model
#         self.dataset = dataset
#         self.preprocessor = preprocessor
#         self.postprocessor = postprocessor
#         self.saver = saver
#         self.mode = mode
#         self.evaluator = evaluator
#         self._validate_components()
#
#     def build(self):
#         """构建工作流"""
#         pass
#
#     def _validate_components(self):
#         """验证组件兼容性"""
#         if self.mode == 'evaluation' and not self.evaluator:
#             raise ValueError("Evaluation mode requires evaluator")
#
#     def run_batch(self, batch_data: Any) -> Dict[str, Any]:
#         """执行单批次处理"""
#         raw_inputs = batch_data['data']
#         raw_targets = batch_data.get('targets')
#
#         processed = self.preprocessor(raw_inputs)
#         raw_output = self.model.run(processed)
#         parsed_results = self.postprocessor(raw_output)
#
#         self.saver.save_batch({
#             'inputs': raw_inputs,
#             'predictions': parsed_results,
#             'targets': raw_targets
#         }, batch_id=id(batch_data))
#
#         # 评估计算
#         batch_metrics = {}
#         if self.mode == 'evaluation' and raw_targets is not None:
#             batch_metrics = self.evaluator.compute_batch(
#                 preds=parsed_results,
#                 targets=raw_targets
#             )
#
#         return {
#             'predictions': parsed_results,
#             'metrics': batch_metrics
#         }
#
#     def run(self, *args, **kwargs) -> Dict[str, Any]:
#         """执行完整流程"""
#         all_metrics = []
#
#         for batch in self.dataset:
#             result = self.run_batch(batch)
#             if self.mode == 'evaluation':
#                 all_metrics.append(result['metrics'])
#
#         self.saver.finalize()
#
#         return {
#             'final_metrics': self.evaluator.aggregate(all_metrics) if self.mode == 'evaluation' else None,
#             'save_path': str(self.saver.output_dir)
#         }
#
#     def __call__(self, *args, **kwargs):
#         return self.run(*args, **kwargs)
#
#
# class ClassificationResultSaver(BaseResultSaver):
#     """分类结果保存器"""
#     def __init__(self, output_dir: str, file_format: str = 'csv'):
#         super().__init__(output_dir)
#         self.format = file_format
#         self.temp_data = []
#
#     def save_batch(self, batch_data: Dict, batch_id: int):
#         for img, pred, target in zip(batch_data['inputs'], batch_data['predictions'], batch_data.get('targets', [])):
#             record = {
#                 'image_path': img.filename if hasattr(img, 'filename') else 'unknown',
#                 'pred_class': pred['class_name'],
#                 'confidence': pred['confidence'],
#                 'true_class': target['class_name'] if target else None
#             }
#             self.temp_data.append(record)
#
#     def finalize(self):
#         if self.format == 'csv':
#             with open(self.output_dir / 'results.csv', 'w') as f:
#                 writer = csv.DictWriter(f, fieldnames=self.temp_data[0].keys())
#                 writer.writeheader()
#                 writer.writerows(self.temp_data)
#         elif self.format == 'json':
#             with open(self.output_dir / 'results.json', 'w') as f:
#                 json.dump(self.temp_data, f)
#
#
# class DetectionResultSaver(BaseResultSaver):
#     """检测结果保存器（COCO格式）"""
#     def __init__(self, output_dir: str):
#         super().__init__(output_dir)
#         self.results = []
#         self.coco_template = {
#             "info": {...},
#             "licenses": [...],
#             "categories": [...],
#             "images": [],
#             "annotations": []
#         }
#
#     def save_batch(self, batch_data: Dict, batch_id: int):
#         for img, pred in zip(batch_data['inputs'], batch_data['predictions']):
#             image_entry = {
#                 "id": len(self.coco_template['images']) + 1,
#                 "file_name": img.filename,
#                 "width": img.width,
#                 "height": img.height
#             }
#             self.coco_template['images'].append(image_entry)
#
#             for box in pred['detections']:
#                 annotation = {
#                     "id": len(self.coco_template['annotations']) + 1,
#                     "image_id": image_entry['id'],
#                     "category_id": box['class_id'],
#                     "bbox": [box['xmin'], box['ymin'], box['xmax']-box['xmin'], box['ymax']-box['ymin']],
#                     "score": box['confidence']
#                 }
#                 self.coco_template['annotations'].append(annotation)
#
#     def finalize(self):
#         with open(self.output_dir / 'detections.json', 'w') as f:
#             json.dump(self.coco_template, f)


class Workflow:
    def __init__(self, tasks):
        self.tasks = {}          # 存储所有任务实例（id -> Task）
        self.successors = defaultdict(list)  # 存储任务的后继节点（用于拓扑排序）
        self.sorted_tasks = []   # 拓扑排序后的任务执行顺序

        # 将依赖的字符串 ID 转换为 Task 实例
        for task in self.tasks.values():
            task.dependencies = [self.tasks[dep_id] for dep_id in task.dependencies]

        # 3. 构建后继节点映射（用于拓扑排序）
        for task in self.tasks.values():
            for dependency in task.dependencies:
                self.successors[dependency].append(task)

        # 4. 执行拓扑排序，验证 DAG 并确定执行顺序
        self.sorted_tasks = self._topological_sort()

    def _topological_sort(self):
        """通过 Kahn 算法实现拓扑排序，返回有序任务列表"""
        in_degree = {task: len(task.dependencies) for task in self.tasks.values()}
        queue = deque([task for task, degree in in_degree.items() if degree == 0])
        sorted_tasks = []

        while queue:
            current_task = queue.popleft()
            sorted_tasks.append(current_task)
            for successor in self.successors[current_task]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(sorted_tasks) != len(self.tasks):
            raise ValueError("Workflow contains a cycle. Not a valid DAG!")
        return sorted_tasks

    def run(self):
        """按拓扑顺序依次执行任务"""
        print("Starting workflow execution...")
        for task in self.sorted_tasks:
            # 检查依赖是否均已执行（理论上拓扑排序已保证）
            for dependency in task.dependencies:
                if not dependency.has_run:
                    raise RuntimeError(f"Task {task.id}'s dependency {dependency.id} not run!")
            # 执行当前任务
            print(f"> Starting task {task.id}")
            task.run()
            print(f"> Completed task {task.id}\n")
        print("All tasks executed successfully!")


class WorkflowBuilder(Builder):
    def __init__(self):
        super().__init__()

    def build(self):
        tasks = []
        return tasks
