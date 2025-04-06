import io
from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple, Dict, Any
from PIL import Image

import numpy as np

from . import torch
from AITools.utils.compatibility import StrEnum

__all__ = [
    "IMG_FORMATS", 
    "VID_FORMATS", 
    "BoxFormat", 
    "ImageType", 
    "ImageData",
    "BoundingBox", 
    "DetectionLabel", 
    "KeypointLabel", 
    "ClassificationLabel", 
    "SegmentationLabel",
    "OCRLabel", 
    "MaskFormat", 
    "InstanceSegmentationLabel",
    "DataItem"
]


IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
VID_FORMATS = ["mp4", "mov", "avi", "mkv"]


@dataclass
class BoxFormat(StrEnum):
    XYXY = "xyxy"         # left top and right bottom
    LTRB = "ltrb"         # left top and right bottom
    LTWH = "ltwh"         # left top and width, height(COCO format)
    XYWH = "xywh"         # center x, center y and width, height
    XYWHA = "xywha"       # center x, center y, width, height and angle
    YOLO = "yolo"         # center x, center y and width, height normalized(YOLO det format)
    YOLO_OBB = "yoloobb"  # center x, center y, width, height and angle normalized(YOLO obb format)
    COCO = "coco"         # left top and width, height(COCO format)
    POINTS = "points"     # points


@dataclass
class ImageType(StrEnum):
    """图像数据存储模式"""
    NUMPY_ARRAY = "numpy"  # 内存中的numpy数组（默认）
    TENSOR = "tensor"      # PyTorch/TensorFlow等框架张量
    FILE_PATH = "path"     # 文件路径（延迟加载）
    BYTE_STREAM = "bytes"  # 字节流（如从网络接收）
    PIL_IMAGE = "pil"      # PIL.Image对象
    LAZY_LOADER = "lazy"   # 自定义延迟加载函数


@dataclass
class ImageFormat(StrEnum):
    """图像数据存储模式"""
    RGB = "rgb"      # RGB格式（默认）
    BGR = "bgr"      # BGR格式
    GRAY = "gray"    # 灰度图
    RGBA = "rgba"    # RGBA格式
    CMYK = "cmyk"    # CMYK格式
    YUV = "yuv"      # YUV格式
    HSV = "hsv"      # HSV格式
    LAB = "lab"      # LAB格式
    XYZ = "xyz"      # XYZ格式
    YCrCb = "ycrcb"  # YCrCb格式


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
    format: str = "RGB"       # 颜色通道格式
    shape: tuple = None       # 图像尺寸
    id: Optional[str] = None  # 唯一标识符
    input_name: Optional[str] = None  # 输入节点名称

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
        """Unified conversion to numpy array"""
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
            self._decoded = self.data()  # Execute the lazy load function

        return self._decoded

    def to_tensor(self, device="cpu") -> torch.Tensor:
        """Convert to PyTorch tensor"""
        return torch.from_numpy(self.to_numpy()).to(device)

    def to_pil(self) -> Image.Image:
        """Convert to PIL image"""
        return Image.fromarray(self.to_numpy())


@dataclass
class BoundingBox:
    """通用边界框定义（支持多种格式）"""
    coords: Union[
        List[Union[int, float]],       # 一维列表，如 [x1, y1, x2, y2] 或 [x, y, w, h]
        List[List[Union[int, float]]]  # 二维列表，如 [[x1, y1], [x2, y2]]
    ]
    format: str                        # 格式标识，如 "xyxy" | "xywh" | "yolo"
    normalized: bool = False           # 是否归一化坐标（相对于图像尺寸）


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
    class_confidence: Optional[float] = None  # 预测时使用


@dataclass
class SegmentationLabel:
    """语义分割标签"""
    mask: np.ndarray                            # 2D或3D的掩码
    class_map: Optional[Dict[int, str]] = None  # 类别ID到名称的映射
    score_map: Optional[np.ndarray] = None      # 分数图（可选）


@dataclass
class OCRLabel:
    """OCR标签"""
    text: str
    text_box: BoundingBox = None                  # 文本区域边界框
    text_direction: int = None                    # 暂定文本旋转角度
    char_boxes: List[BoundingBox] = None          # 字符级边界框
    text_confidence: Optional[float] = None       # 预测时使用
    box_confidence: Optional[float] = None        # 预测时使用
    direction_confidence: Optional[float] = None  # 预测时使用


@dataclass
class MaskFormat(StrEnum):
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
class DataItem:
    """数据集返回的原子单元"""
    data: Dict[str, Union[
        ImageData,
        Any
    ]] = field(default_factory=dict)
    labels: List[Union[
        DetectionLabel,
        ClassificationLabel,
        SegmentationLabel,
        OCRLabel,
        KeypointLabel,
        InstanceSegmentationLabel,
        Any
    ]] = field(default_factory=list)
    predictions: List[Union[
        DetectionLabel,
        ClassificationLabel,
        SegmentationLabel,
        OCRLabel,
        KeypointLabel,
        InstanceSegmentationLabel,
        Any
    ]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiIOConfig:
    preprocessors: Dict[str, List[Any]] = field(default_factory=dict)
    postprocessors: Dict[str, List[Any]] = field(default_factory=dict)
    parsers: Dict[str, Any] = field(default_factory=dict)


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
    return DataItem(
        image=ImageData(
            data=...,
            path=image_info['file_name'],
            id=image_info['id']
        ),
        labels=labels
    )



def convert_coco_instance(coco_ann, img_info, img_array):
    item = DataItem(
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


def instances_to_semantic(item: DataItem) -> DataItem:
    """将实例分割转换为语义分割（类别聚合）"""
    h, w = item.image.data.shape[:2]
    semantic_mask = np.zeros((h, w), dtype=np.int32)

    for label in item.labels:
        if isinstance(label, InstanceSegmentationLabel):
            # 获取该实例的二值掩码
            instance_mask = MaskDecoder.to_binary_mask(label, (h, w))
            # 将对应区域设为类别ID
            semantic_mask[instance_mask == 1] = label.class_id

    return DataItem(
        image=item.image,
        labels=[SegmentationLabel(mask=semantic_mask)],
        metadata=item.metadata
    )
