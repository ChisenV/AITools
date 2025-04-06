import os.path
from pathlib import Path

from AITools import OCRDatasetV2

abs_file = r"E:\python_ai_dataset\train\Annotations\PinHole@201@201@1@pin0@ID27417(63.44)-NG.xml"
rel_file = r"train\Annotations\PinHole@201@201@1@pin0@ID27417(63.44)-NG.xml"


def path_test():
    print(Path(rel_file))

    print(os.path.normpath(rel_file).split(os.sep))
    print(os.path.join(*os.path.normpath(rel_file).split(os.sep)[1:]))
    print(os.path.dirname(abs_file))


def det_paths(path=r"E:\python_ai_dataset\OCR\det\gather\categories"):
    return [os.path.join(path, i) for i in os.listdir(path)
            if os.path.isdir(os.path.join(path, i))]


def rec_paths(top_dir=r"E:\python_ai_dataset\OCR\rec\gather\categories"):
    return [os.path.join(top_dir, i) for i in os.listdir(top_dir)
            if os.path.isdir(os.path.join(top_dir, i))]


def OCRDatesetV2_init_case1():
    d = OCRDatasetV2()
    assert len(d) == 0
    print(d.__dict__)


def OCRDatesetV2_init_case2():
    pass


def OCRDatesetV2_init_case3():
    pass


def test_OCRDatesetV2_init():
    OCRDatesetV2_init_case1()
    OCRDatesetV2_init_case2()
    OCRDatesetV2_init_case3()
