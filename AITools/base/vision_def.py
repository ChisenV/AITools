import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, List, Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

from AITools.core import manager

try:
    import torch
except ImportError:
    class torch:
        Tensor = None

        @staticmethod
        def from_numpy(*args, **kwargs):
            raise ImportError("PyTorch is not installed. Please install it to use this feature.")

__all__ = [
    "IMG_FORMATS", "VID_FORMATS", "BoxFormat", "ImageType", "ImageData",
    "BoundingBox", "DetectionLabel", "KeypointLabel", "ClassificationLabel", "SegmentationLabel",
    "OCRLabel", "MaskFormat", "InstanceSegmentationLabel", "DatasetItem"
]

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]


@dataclass
class BoxFormat(Enum):
    XYXY = "xyxy"         # left top and right bottom
    LTRB = "ltrb"         # left top and right bottom
    LTWH = "ltwh"         # left top and width, height(COCO format)
    XYWH = "xywh"         # center x, center y and width, height
    XYWHA = "xywha"       # center x, center y, width, height and angle
    YOLO = "yolo"         # center x, center y and width, height normalized(YOLO det format)
    YOLO_OBB = "yoloobb"  # center x, center y, width, height and angle normalized(YOLO obb format)
    COCO = "coco"         # left top and width, height(COCO format)


@dataclass
class ImageType(Enum):
    """图像数据存储模式"""
    NUMPY_ARRAY = "numpy"  # 内存中的numpy数组（默认）
    TENSOR = "tensor"      # PyTorch/TensorFlow等框架张量
    FILE_PATH = "path"     # 文件路径（延迟加载）
    BYTE_STREAM = "bytes"  # 字节流（如从网络接收）
    PIL_IMAGE = "pil"      # PIL.Image对象
    LAZY_LOADER = "lazy"   # 自定义延迟加载函数


@dataclass
class ImageData:
    data: Union[
        np.ndarray,    # 直接存储numpy数组
        torch.Tensor,  # PyTorch张量
        str,           # 文件路径/URL
        bytes,         # 字节流
        Image.Image,   # PIL对象
        callable,      # 延迟加载函数
    ]
    path: str
    format: str = "RGB"  # 颜色通道格式
    id: Optional[str] = None  # 唯一标识符

    # ---------- 类型元数据（自动推断） ----------
    _type: ImageType = field(init=False)
    _decoded: Optional[Any] = field(default=None, init=False)  # 解码缓存

    def __post_init__(self):
        """自动推断数据类型"""
        if isinstance(self.data, np.ndarray):
            self._type = ImageType.NUMPY_ARRAY
        elif isinstance(self.data, torch.Tensor):
            self._type = ImageType.TENSOR
        elif isinstance(self.data, str):
            self._type = ImageType.FILE_PATH
        elif isinstance(self.data, bytes):
            self._type = ImageType.BYTE_STREAM
        elif isinstance(self.data, Image.Image):
            self._type = ImageType.PIL_IMAGE
        elif callable(self.data):
            self._type = ImageType.LAZY_LOADER
        else:
            raise TypeError(f"Unsupported image data type: {type(self.data)}")

    # ---------- 核心方法：按需转换 ----------
    def to_numpy(self) -> np.ndarray:
        """统一转换为numpy数组"""
        if self._decoded is not None:
            return self._decoded

        if self._type == ImageType.NUMPY_ARRAY:
            self._decoded = self.data
        elif self._type == ImageType.TENSOR:
            self._decoded = self.data.cpu().numpy()
        elif self._type == ImageType.FILE_PATH:
            self._decoded = np.array(Image.open(self.data))
        elif self._type == ImageType.BYTE_STREAM:
            self._decoded = np.array(Image.open(io.BytesIO(self.data)))
        elif self._type == ImageType.PIL_IMAGE:
            self._decoded = np.array(self.data)
        elif self._type == ImageType.LAZY_LOADER:
            self._decoded = self.data()  # 执行延迟加载函数

        return self._decoded

    def to_tensor(self, device="cpu") -> torch.Tensor:
        """转换为PyTorch张量"""
        return torch.from_numpy(self.to_numpy()).to(device)

    def to_pil(self) -> Image.Image:
        """转换为PIL图像"""
        return Image.fromarray(self.to_numpy())


@manager.DATASETS.register_component
@dataclass
class BoundingBox:
    """通用边界框定义（支持多种格式）"""
    coords: List[float]         # 坐标值
    format: str                 # 格式标识，如 "xyxy" | "xywh" | "yolo"
    normalized: bool = False    # 是否归一化坐标（相对于图像尺寸）


@dataclass
class DetectionLabel:
    """目标检测标签"""
    bbox: BoundingBox
    class_id: int
    class_name: Optional[str] = None
    confidence: Optional[float] = None  # 预测时使用
    is_crowd: bool = False              # COCO兼容


@dataclass
class KeypointLabel:
    points: List[Tuple[float, float]]  # 关键点坐标
    visible: List[bool]                # 可见性标识
    format: str = "absolute"           # 坐标格式


@dataclass
class ClassificationLabel:
    """图像分类标签"""
    class_id: int
    class_name: Optional[str] = None


@dataclass
class SegmentationLabel:
    """语义分割标签"""
    mask: np.ndarray                            # 2D或3D的掩码
    class_map: Optional[Dict[int, str]] = None  # 类别ID到名称的映射


@dataclass
class OCRLabel:
    """OCR标签"""
    text: str
    char_boxes: List[BoundingBox]    # 字符级边界框
    text_box: BoundingBox            # 文本区域边界框
    text_direction: int              # 暂定文本旋转角度


@dataclass
class MaskFormat(Enum):
    BINARY = "binary"       # 二值掩码 (0/1)
    INDEXED = "indexed"     # 实例ID索引 (0~N)
    RLE = "rle"             # COCO的Run-Length Encoding
    POLYGON = "polygon"     # 多边形坐标点


@dataclass
class InstanceSegmentationLabel:
    """实例分割标签（继承检测标签扩展）"""
    bbox: BoundingBox               # 继承目标检测的边界框
    mask: Union[
        np.ndarray,                 # 支持numpy数组
        List[List[float]],          # 多边形坐标 [[x1,y1,x2,y2,...]]
        dict                        # RLE字典格式
    ]
    mask_format: MaskFormat         # 明确标注掩码格式
    class_id: int
    instance_id: Optional[int] = None  # 实例唯一标识（用于跟踪任务）
    confidence: Optional[float] = None


@dataclass
class DatasetItem:
    """数据集返回的原子单元"""
    image: ImageData
    labels: List[Union[
        DetectionLabel,
        ClassificationLabel,
        SegmentationLabel,
        OCRLabel,
        KeypointLabel,
        InstanceSegmentationLabel,
        Any
    ]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def parse_yolo_label(image_path, label_path):
    with open(label_path) as f:
        labels = []
        for line in f:
            class_id, x_center, y_center, w, h = map(float, line.split())
            bbox = BoundingBox(
                coords=[x_center, y_center, w, h],
                format="yolo",
                normalized=True
            )
            labels.append(DetectionLabel(bbox=bbox, class_id=int(class_id)))
        return DatasetItem(
            image=ImageData(data=..., path=image_path),
            labels=labels
        )


def coco_to_protocol(coco_ann, image_info):
    labels = []
    for ann in coco_ann:
        bbox = BoundingBox(
            coords=ann['bbox'],  # COCO使用[x,y,w,h]
            format="xywh"
        )
        labels.append(DetectionLabel(
            bbox=bbox,
            class_id=ann['category_id'],
            is_crowd=ann['iscrowd']
        ))
    return DatasetItem(
        image=ImageData(
            data=...,
            path=image_info['file_name'],
            id=image_info['id']
        ),
        labels=labels
    )


def parse_voc_segmentation(image_path, mask_path):
    return DatasetItem(
        image=ImageData(data=..., path=image_path),
        labels=[
            SegmentationLabel(mask=...)
        ]
    )


def yolo_to_absolute(bbox: BoundingBox, img_width: int, img_height: int):
    if bbox.format != "yolo" or not bbox.normalized:
        return bbox
    x, y, w, h = bbox.coords
    return BoundingBox(
        coords=[
            (x - w/2) * img_width,   # x_min
            (y - h/2) * img_height,  # y_min
            (x + w/2) * img_width,   # x_max
            (y + h/2) * img_height   # y_max
        ],
        format="xyxy",
        normalized=False
    )


def convert_coco_instance(coco_ann, img_info, img_array):
    item = DatasetItem(
        image=ImageData(
            data=img_array,
            path=img_info['file_name'],
            id=img_info['id']
        )
    )

    for ann in coco_ann:
        # 转换边界框 (COCO使用[x,y,width,height])
        bbox = BoundingBox(
            coords=ann['bbox'],
            format="xywh",
            normalized=False
        )

        # 处理掩码格式
        if isinstance(ann['segmentation'], dict):
            mask_data = ann['segmentation']  # RLE
            mask_format = MaskFormat.RLE
        else:
            mask_data = ann['segmentation']  # 多边形列表
            mask_format = MaskFormat.POLYGON

        # 构建实例标签
        instance_label = InstanceSegmentationLabel(
            bbox=bbox,
            mask=mask_data,
            mask_format=mask_format,
            class_id=ann['category_id'],
            instance_id=ann['id']
        )
        item.labels.append(instance_label)

    return item


class MaskDecoder:
    @staticmethod
    def to_binary_mask(label: InstanceSegmentationLabel, img_size: Tuple[int,int]):
        """将不同格式转换为二值掩码"""
        if label.mask_format == MaskFormat.RLE:
            # TODO return coco_mask.decode(label.mask)
            pass
        elif label.mask_format == MaskFormat.POLYGON:
            # TODO return polygon_to_mask(label.mask, img_size)
            pass
        elif label.mask_format == MaskFormat.BINARY:
            return label.mask
        else:
            raise ValueError(f"Unsupported mask format: {label.mask_format}")


def instances_to_semantic(item: DatasetItem) -> DatasetItem:
    """将实例分割转换为语义分割（类别聚合）"""
    h, w = item.image.data.shape[:2]
    semantic_mask = np.zeros((h, w), dtype=np.int32)

    for label in item.labels:
        if isinstance(label, InstanceSegmentationLabel):
            # 获取该实例的二值掩码
            instance_mask = MaskDecoder.to_binary_mask(label, (h, w))
            # 将对应区域设为类别ID
            semantic_mask[instance_mask == 1] = label.class_id

    return DatasetItem(
        image=item.image,
        labels=[SegmentationLabel(mask=semantic_mask)],
        metadata=item.metadata
    )
