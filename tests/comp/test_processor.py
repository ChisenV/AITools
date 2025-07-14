import os
import random

import cv2
import numpy as np

from AITools import IMG_FORMATS
from AITools.comp.processor import MosaicImage
from AITools.comp.functions import imread, imshow, compute_affine_matrix, imwrite


def test_get_level():
    # imh, imw = 1121, 801
    # level_table = [i * 320 for i in range(0, 15)]
    # print(level_table)
    # print(imh/level_table[-1] / 2 * 10, imw/level_table[-1] / 2 * 10)
    # print(round(imh / level_table[-1] / (10 / (len(level_table)-1)) * 10, 0),
    #       round(imw / level_table[-1] / (10 / (len(level_table)-1)) * 10, 0))
    #
    # print(int(round(imh / level_table[-1] * (len(level_table) - 1), 0)),
    #       int(round(imw / level_table[-1] * (len(level_table) - 1), 0)))

    img = imread(r"E:\python_ai_dataset\foreign-object-detect\2025\异物\异物-AIDIAN\bead\image11 (3).jpeg")
    imh, imw = img.shape[0], img.shape[1]
    imshow("tf", img)

    class PixelRuler:
        def __init__(self, step, length):
            self.step = step
            self.length = length
            self.ruler = np.arange(start=0, stop=length, step=step)

        def get_level(self, value):
            return int(round(value / self.step, 0))

        def get_ruler(self):
            return self.ruler

        def get_measure(self, value):
            index = self.get_level(value)
            if index > len(self.ruler) - 1:
                raise ValueError("Value is overflow")
            return self.ruler[index]

        def reset(self, step, length):
            self.step = step
            self.length = length
            self.ruler = np.arange(start=0, stop=length, step=step)

    ruler = PixelRuler(320, 15000)
    print(ruler.get_ruler())
    print(ruler.get_measure(imh), ruler.get_measure(imw), ruler.get_level(32000))
    to_size = max(ruler.get_measure(imh), ruler.get_measure(imw))
    print(to_size)
    src2dst, dst2src = compute_affine_matrix((imw, imh), (to_size, to_size))
    print(src2dst, dst2src)
    af_img = cv2.warpAffine(img,
                            src2dst,
                            (to_size, to_size),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(114, 114, 114))
    imshow("af", af_img, 0)


def test_MosaicImage():
    m = MosaicImage(r"E:\python_ai_dataset\foreign-object-detect\2025\异物-anno\异物-AIDIAN")
    print(len(m.image_paths))
    for i in m.image_paths:
        print(i)


def test_cvRotate():
    im = np.zeros([300, 300, 3], dtype=np.uint8)

    w, h = 80, 120

    rrt = cv2.RotatedRect((150, 150), (w, h), 90.12733)
    cv2.polylines(im, [rrt.points().astype(np.int32)], True, (255, 255, 255), 2)
    print(rrt.angle)
    angle = np.deg2rad(rrt.angle)
    angle = angle - np.pi / 2 if angle != np.pi / 2 else angle - np.pi  # oc' -> oc
    w, h = (h, w) if angle != np.pi / 2 else (w, h)  # oc' -> oc
    angle = angle if w > h else angle + np.pi / 2  # oc -> le
    w, h = (w, h) if w > h else (h, w)  # oc -> le
    angle = angle + np.pi if - np.pi / 2 <= angle < -np.pi / 4 else angle  # le -> le'
    print(angle, np.rad2deg(angle))

    rrt = cv2.RotatedRect((150, 150), (w, h), -70)
    cv2.polylines(im, [rrt.points().astype(np.int32)], True, (255, 0, 0), 2)

    rrt = cv2.RotatedRect((150, 150), (w, h), np.rad2deg(angle))
    cv2.polylines(im, [rrt.points().astype(np.int32)], True, (0, 250, 0), 2)

    imshow("im", im, 0)


def regularize_rboxes(rboxes: np.ndarray):
    """
    Regularize rotated boxes in range [0, pi/2].

    Args:
        rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.

    Returns:
        (torch.Tensor): The regularized boxes.
    """
    x, y, w, h, t = np.unstack(rboxes, axis=-1)
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = t % np.pi >= np.pi / 2
    w_ = np.where(swap, h, w)
    h_ = np.where(swap, w, h)
    t = t % (np.pi / 2)
    return np.stack([x, y, w_, h_, t], axis=-1)  # regularized boxes

def regularize_rboxes_135(rboxes: np.ndarray):
    w, h, angle = rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    angle = angle - np.pi / 2 if angle != np.pi / 2 else angle - np.pi  # oc' -> oc
    w, h = (h, w) if angle != np.pi / 2 else (w, h)  # oc' -> oc
    angle = angle if w > h else angle + np.pi / 2  # oc -> le
    w, h = (w, h) if w > h else (h, w)  # oc -> le
    angle = angle + np.pi if - np.pi / 2 <= angle < -np.pi / 4 else angle  # le -> le'


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


# rbox function implemented using numpy
def poly2rbox_le135_np(poly):
    """convert poly to rbox [-pi / 4, 3 * pi / 4]

    Args:
        poly: [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        rbox: [cx, cy, w, h, angle]
    """

    def per(x):
        cx, cy, w, h, angle = x
        p = cv2.RotatedRect((cx, cy), (w, h), angle).points().reshape(-1)
        return p

    y = np.apply_along_axis(per, axis=1, arr=poly)
    poly = np.array(y[:, :8], dtype=np.float32)

    pt1 = (poly[:, 0], poly[:, 1])
    pt2 = (poly[:, 2], poly[:, 3])
    pt3 = (poly[:, 4], poly[:, 5])
    pt4 = (poly[:, 6], poly[:, 7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))

    width = np.maximum(edge1, edge2)
    height = np.minimum(edge1, edge2)

    # rbox_angle = 0
    # if edge1 > edge2:
    #     rbox_angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    # elif edge2 >= edge1:
    #     rbox_angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    rbox_angle = np.where(
        edge1 > edge2,
        np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]),
        np.arctan2(pt4[1] - pt1[1], pt4[0] - pt1[0])
    )
    rbox_angle = norm_angle(rbox_angle)

    x_ctr = (pt1[0] + pt3[0]) / 2
    y_ctr = (pt1[1] + pt3[1]) / 2
    return np.stack([x_ctr, y_ctr, width, height, rbox_angle], axis=-1)


def regularize_rboxes2(rboxes: np.ndarray):  # le' -> oc'
    x, y, w, h, t = np.unstack(rboxes, axis=-1)
    t_le = np.where(np.logical_and(t >= np.pi / 2, t < 3 * np.pi / 4),
                       t - np.pi, t)
    t_oc = np.where(np.logical_and(t_le >= -np.pi / 2, t_le < 0.),
                       t_le, t_le - np.pi / 2)
    w_oc = np.where(np.logical_and(t_le >= -np.pi / 2, t_le < 0.),
                       w, h)
    h_oc = np.where(np.logical_and(t_le >= -np.pi / 2, t_le < 0.),
                       h, w)
    t = np.where(t_oc != -np.pi / 2, t_oc + np.pi / 2, t_oc + np.pi)
    w_ = np.where(t_oc != -np.pi / 2, h_oc, w_oc)
    h_ = np.where(t_oc != -np.pi / 2, w_oc, h_oc)
    t = np.where(t >= np.pi / 2, t % (np.pi / 2), t)
    t = np.where(t == 0., t + (np.pi / 2), t)
    return np.stack([x, y, w_, h_, t], axis=-1)  # regularized boxes

def ocr2ler(rboxes: np.ndarray):
    x, y, w, h, t = np.unstack(rboxes, axis=-1)
    # t = t - np.pi / 2 if t != np.pi / 2 else t - np.pi  # oc' -> oc
    t_oc = np.where(t != np.pi / 2, t - np.pi / 2, t - np.pi)
    # w, h = (h, w) if t != np.pi / 2 else (w, h)  # oc' -> oc
    w_oc = np.where(t != np.pi / 2, h, w)
    h_oc = np.where(t != np.pi / 2, w, h)
    # t = t if w > h else t + np.pi / 2  # oc -> le
    t_le = np.where(w_oc > h_oc, t_oc, t_oc + np.pi / 2)
    # w, h = (w, h) if w > h else (h, w)  # oc -> le
    w_le = np.where(w_oc > h_oc, w_oc, h_oc)
    h_le = np.where(w_oc > h_oc, h_oc, w_oc)
    # t = t + np.pi if - np.pi / 2 <= t < -np.pi / 4 else t  # le -> le'
    t = np.where(np.logical_and(- np.pi / 2 <= t_le, t_le < -np.pi / 4), t_le + np.pi, t_le)
    return np.stack([x, y, w_le, h_le, t], axis=-1)  # regularized boxes


def test_regularize_rboxes_comp():
    print()
    for deg in range(-45, 136, 1):
        rad = np.deg2rad(deg)
        rboxes = np.array(
            [[150, 150, 100, 120, rad], [150, 150, 100, 100, rad], [150, 150, 120, 100, rad]]
        )
        rrb1 = regularize_rboxes(rboxes)
        rrb2 = regularize_rboxes2(rboxes)

        for i, (r1, r2) in enumerate(zip(rrb1, rrb2)):
            x1, y1, w1, h1, t1 = r1
            x2, y2, w2, h2, t2 = r2

            if (abs(t1 - t2) > 1e-6 or t1 > np.pi / 2 or t2 > np.pi / 2 or t1 < 0 or t2 < 0
                    or deg == 134 or deg == 1 or deg == 89 or deg == 90 or deg == 45 or deg == -45 or deg == -1):
                # print(f"{deg} {i}: {x1}, {y1}, {w1}, {h1}, {t1} != {x2}, {y2}, {w2}, {h2}, {t2}")
                pass

        if deg > 0 and deg <= 90:
            # if deg == 1 or deg == 45 or deg == 89 or deg == 90:
            #     print(f"B================================")
            rrb3 = ocr2ler(rboxes)
            # rrb4 = poly2rbox_le135_np(rboxes)
            for i, r3 in enumerate(rrb3):
                if i != 0:
                    continue
                x3, y3, w3, h3, t3 = r3
                print(f"{deg} {i}: {x3}, {y3}, {w3}, {h3}, {t3}")
                # print(f"E================================")

    print("ok")


def test_regularize_rboxes():
    print()
    im = np.zeros([300, 300, 3], dtype=np.uint8)
    rboxes = np.array([[150, 150, 119, 120, 0.1], [150, 150, 100, 100, -np.pi/4], [150, 150, 100, 100, np.pi/3]])
    rrb = regularize_rboxes(rboxes)
    print(rrb)
    c = [(255, 0, 0), (0, 0, 250), (0, 250, 0)]
    for idx, i in enumerate(rrb):
        cx, cy = i[0], i[1]
        w, h = i[2], i[3]
        rrt = cv2.RotatedRect((cx, cy), (w, h), np.rad2deg(i[4]))
        cv2.polylines(im, [rrt.points().astype(np.int32)], True, c[idx], 2)
    imshow("im", im, 0)


def test_oc_minAreaRect():
    for i in range(-45, 181, 1):
        rrt = cv2.RotatedRect((150, 150), (120, 121), i)
        (cx, cy), (w, h), angle = cv2.minAreaRect(rrt.points())
        print(i, round(cx), round(cy), round(w), round(h), angle)


def test_rotate_img():
    def rotate_image_around_point(image, center, angle_deg, imgsz):
        """
        以指定点为中心旋转图像，保持图像尺寸不变

        参数:
            image: 输入图像 (numpy数组)
            center: 旋转中心点坐标 (x, y)
            angle_deg: 旋转角度(度)，正值表示逆时针旋转

        返回:
            rotated_image: 旋转后的图像
        """
        # 获取图像尺寸
        (h, w) = image.shape[:2]

        # 计算旋转矩阵
        R = np.eye(3, dtype=np.float32)
        R[:2] = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        # 平移 缩放
        A = np.eye(3, dtype=np.float32)
        A[:2] = compute_affine_matrix((w, h), imgsz)[0]
        rotation_matrix = A @ R

        # 执行旋转（指定输出尺寸为原始尺寸）
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix[:2],
            imgsz,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114))  # 填充黑边

        return rotated_image

    img_path = r"E:\python_ai_dataset\obb\AILocate-V2.8-splits\images\test\3D-SMT_QFN\3D##2##20231222095444##fov_34_0@[0]@[1266,2203]@[a=0] (2).bmp"
    # img_path = r"E:\python_ai_dataset\obb\3D-SMT_IC-std\3D-SMT_IC\JPEGImages\3D##7598TG##20231215095350##fov_23_0@[7]@[732,1227]@[a=0] (2).bmp"
    save_dir = r"E:\python_ai_dataset\obb\360-3"
    os.makedirs(save_dir, exist_ok=True)
    image = imread(img_path)
    for deg in range(0, 360, 1):
        img = rotate_image_around_point(image, (272, 276), deg, (640, 640))
        # img = rotate_image_around_point(image, (335, 540), deg, (640, 640))
        # cv2.circle(img,(272, 276), radius=2, color=(0,0,255))
        # imshow("img", img, 0)
        imwrite(os.path.join(save_dir, f"{deg}.png"), img)


def test_negative_zero():
    f = -0.0
    d = 0.0
    print(f+90, f==d)

    ar = np.array([-1,-2,3,0,4,5])
    pos = ar + 1 < 0
    print(pos)
    ar[pos] = ar[pos] - 6
    print(ar)

    ar2 = np.array([[-1],[-2],[3],[0],[4],[5]])
    print(ar2 * 2, ar * 2)


def test_resize():
    dir_path = r"E:\python_ai_dataset\obb\3D-SMT_IC-square\3D-SMT_IC_splits\images\stretch"
    images = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith(tuple(IMG_FORMATS))]

    for img_path in images:
        img = imread(img_path)
        print(img.shape)
        if random.random() < 0.5:
            img_s = cv2.resize(img, (img.shape[1] // 2, img.shape[0]))
        else:
            img_s = cv2.resize(img, (img.shape[1], img.shape[0]//2))
        img_s = cv2.resize(img_s, (img_s.shape[1] //2, img_s.shape[0] //2))
        img = cv2.resize(img, (img.shape[1] //2, img.shape[0] //2))
        imshow("img", img)
        imshow("img_s", img_s, 0)


def test_phase():
    ns = 3
    for angle in np.linspace(-np.pi/4, 3*np.pi/4, 180):
        phase_angle = angle * 2  # φ1 = 2 * θ, θ ∈ [-π/2, π/2)
        phase_shift_X = tuple(
            np.cos(phase_angle + 2 * np.pi * n / ns).round(6).tolist()  # x_n = cos(φ + 2 * n * π / N_step)
            for n in range(ns)
        )  # X1
        print(phase_angle, phase_shift_X)


class PSCCoder:
    """Phase-Shifting Coder.

    `Phase-Shifting Coder (PSC)
    <https://arxiv.org/abs/2211.06368>`.

    Args:
        ang_ver (str): Angle definition version. Only 'le90'(long-edge 90) is supported at present.
        df (bool, optional): Whether to use dual frequency. Default: True.
        ns (int, optional): Number of phase steps. Also denoted as N_step, Default: 3.
        tm (float): Threshold of modulation. Default: 0.47.
    """
    # The size of the last of dimension of the encoded tensor.
    encode_size = 4

    def __init__(self, ns: int = 3, df: bool = True, tm: float = 0.47, ang_ver: str = 'le90'):
        super().__init__()
        assert ang_ver in ['le90']
        self.df = df
        self.ns = ns
        self.tm = tm
        self.encode_size = 2 * self.ns if self.df else self.ns
        self.ang_ver = ang_ver

        # Note: In the paper, n starts from 1, while in this code, it starts from 0.
        self.coef_sin = np.array(tuple(
            np.sin(np.array(2 * n * np.pi / self.ns))
            for n in range(self.ns)
        ))  # sin(2 * n * π / N_step)
        self.coef_cos = np.array(tuple(
            np.cos(np.array(2 * n * np.pi / self.ns))
            for n in range(self.ns)
        ))  # cos(2 * n * π / N_step)

    def encode(self, angle):
        """Phase-Shifting Encoder.

        Args:
            angle (np.Tensor): Angle offset for each scale level.
                Has shape (H * W, 1)
                Also see 'ultralytics/utils/ops.py': poly2rbox
        Returns:
            list[np.Tensor]: The psc coded data (phase-shifting patterns)
                for each scale level.
                Has shape (H * W, encode_size)
        """
        phase_angle = angle * 2  # φ1 = 2 * θ, θ ∈ [-π/2, π/2)
        phase_shift_X = tuple(
            np.cos(phase_angle + 2 * np.pi * n / self.ns)  # x_n = cos(φ + 2 * n * π / N_step)
            for n in range(self.ns)
        )  # X1

        # Dual-freq PSC for square-like problem
        if self.df:
            phase_angle = angle * 4  # φ2 = 4 * θ, θ ∈ [-π/2, π/2)
            phase_shift_X += tuple(
                np.cos(phase_angle + 2 * np.pi * n / self.ns)
                for n in range(self.ns)
            )  # X2

        return np.concatenate(phase_shift_X, axis=-1)  # {X1, X2}

    def decode(self, x, keepdim: bool = False, ne: int = 1):
        """Phase-Shifting Decoder.

        Args:
            x (np.Tensor): The psc coded data (phase-shifting patterns), angle of prediction.
                for each scale level.
                Has shape (bs, encode_size, L)
            keepdim (bool): Whether the output tensor has dim retained or not.
            ne (int): number of extra parameters, see OBB head

        Returns:
            list[Tensor]: Angle offset for each scale level.
                Has shape (num_anchors * H * W, 1) when keepdim is true,
                (num_anchors * H * W) otherwise
        """

        # Adjust the input dimension: (bs, encode_size, L) -> (bs, L, encode_size)
        # x = x.permute(0, 2, 1)  # Rearrange the dimensions: Place the feature dimensions at the end
        x = np.transpose(x, (0, 2, 1))

        # x = np.sigmoid(x)  # Apply sigmoid on the feature dimension (encode_size)
        batch_size, L = x.shape[0], x.shape[1]  # Save the original batch size and the length of the space dimension

        # Merge batch and spatial dimensions to adapt to the original decoding logic
        x = x.reshape(-1, x.shape[-1])  # new shape: (bs * L, encode_size)
        # self.coef_sin = self.coef_sin.to(x)
        # self.coef_cos = self.coef_cos.to(x)

        # decode φ1 | sum(): Generally, the dimensions should remain unchanged, so keepdim=True
        phase_sin = np.sum(x[:, 0:self.ns] * self.coef_sin, axis=-1, keepdims=keepdim)
        phase_cos = np.sum(x[:, 0:self.ns] * self.coef_cos, axis=-1, keepdims=keepdim)
        phase_mod = phase_cos**2 + phase_sin**2
        phase1 = -np.atan2(phase_sin, phase_cos)  # φ1 ∈ [-pi, pi)

        if self.df:
            # decode φ2
            phase_sin = np.sum(x[:, self.ns:(2 * self.ns)] * self.coef_sin, axis=-1, keepdims=keepdim)
            phase_cos = np.sum(x[:, self.ns:(2 * self.ns)] * self.coef_cos, axis=-1, keepdims=keepdim)
            phase_mod = phase_cos**2 + phase_sin**2
            phase2 = -np.atan2(phase_sin, phase_cos) / 2

            # Phase unwrapping, dual freq mixing, mix them to obtain the final phase
            # Angle between phase1 and phase2 is obtuse angle
            # δ = cos(φ1) * cos(φ2) + sin(φ1) * sin(φ2)
            idx = (np.cos(phase1) * np.cos(phase2) + np.sin(phase1) * np.sin(phase2)) < 0
            # Add pi to phase2 and keep it in range [-pi, pi)
            # print(phase2, phase2[idx] % (2 * np.pi), (phase2 % (2 * np.pi)) - np.pi)
            phase2[idx] = phase2[idx] % (2 * np.pi) - np.pi
            # print(phase2)
            phase1 = phase2

        # Set the angle of isotropic objects to zero
        phase1[phase_mod < self.tm] *= 0  # Force it to be set in the horizontal direction
        _angle = phase1 / 2  # θ = φ / 2

        if keepdim:
            angle = _angle.reshape((batch_size, ne, L))  # shape: (bs, 1, L)
        else:
            angle = _angle.reshape((batch_size, L))  # shape: (bs, L)

        return angle  # angle of prediction, θ ∈ [-π/2, π/2)


def test_coder():
    print()
    coder = PSCCoder(ns=3, df=True)
    for angle in np.linspace(-np.pi / 2, np.pi / 2, 180):
        encode = coder.encode(angle.reshape(-1, 1))
        encode = np.array(encode)[:, :, np.newaxis]
        # print(encode, encode.shape)
        decode = coder.decode(encode, keepdim=True)
        print(angle, "decode:", decode.tolist(), "encode:", encode.tolist())


def test_p():
    # print(cv2.RotatedRect((10, 10), (4, 6), 0).points())
    path = r"E:\python_ai_dataset\FindBoardLebel\V1\JPEGImages"
    dst = r"E:\python_ai_dataset\FindBoardLebel\V1\JPEGImages_2"
    to_size = (640, 640)
    kernel_size = 13  # 确保奇数
    for i, file in enumerate(os.listdir(path)):
        image_path = os.path.join(path, file)
        if not image_path.endswith(tuple(IMG_FORMATS)):
            continue
        basename = os.path.basename(image_path)
        img = imread(image_path)
        h, w = img.shape[:2]
        img = img[500:h-500, 500:w-500]
        rh, rw = img.shape[:2]

        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0)
        s2d, d2s = compute_affine_matrix((rw, rh), to_size)

        # 执行旋转（指定输出尺寸为原始尺寸）
        img = cv2.warpAffine(
            img,
            s2d,
            to_size,
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114))  # 填充黑边
        # img = cv2.resize(img, (640, 640))

        imshow("im", img, 0)
        # imwrite(os.path.join(dst, basename), Affineimage)


def test_ocv():

    print(4.5230e-01)
