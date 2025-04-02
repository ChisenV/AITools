from AITools import Config, COMPONENT_MANAGERS
from AITools.core import Builder


def test_builder_mount_properties():
    print()

    class Mount1:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class Mount2:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class Mount3:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class Mount4:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class Mount5:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class Mount6:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

    class Manager:
        components_dict = {
            "mount1": Mount1,
            "mount2": Mount2,
            "mount3": Mount3,
            "mount4": Mount4,
            "mount5": Mount5,
            "mount6": Mount6,
        }
        name = "c2"

    # 组件管理器
    MOUNT_MANAGER = [Manager()]

    config = {
        "cc": 6,
        "c0": "name",
        "c1": {
            "type": "mount1",
            "param1": 1,
            "param2": 2,
            "param3": 3,
        },
        "c2": {
            "type": "mount2",
            "param1": {
                "type": "mount3",
                "param1": 1,
                "param2": 2,
            },
            "param2": 2,
            "param3": 3,
        },
        "c3": {
            "type": "mount2",
            "param1": {
                "type": "mount4",
                "param1": 1,
                "param2": 2,
            },
            "param2": 2,
            "param3": [
                {
                    "type": "mount5",
                    "param1": 1,
                    "param2": 2,
                },
                {
                    "type": "mount6",
                    "param1": 1,
                    "param2": 2,
                }
            ],
            "c2": {
                "type": "mount2",
                "param1": {
                    "type": "mount3",
                    "param1": 1,
                    "param2": 2,
                },
                "param2": 2,
                "param3": 3,
            }
        },
    }

    def hook1(ins, cfg):
        print(f"hook1: {ins}, {cfg}")

    def test_scope():
        builder = Builder(config, components=MOUNT_MANAGER, name="TrainBuilder", mark="type",
                          post_build_hooks=[hook1])
        print(builder.name)

        for i, (k, v) in enumerate(builder.__dict__.items()):
            print(f"{i}. {k} = {v}")

        place1_c1 = builder.c1
        place1_c2_param1 = builder.c2.param1
        print(place1_c1)
        print(place1_c2_param1)
        assert isinstance(place1_c1, Mount1)  # True
        assert isinstance(place1_c2_param1, Mount3)  # True
        assert isinstance(builder.c3.param3[0], Mount5)

        for i, (k, v) in enumerate(builder.__dict__.items()):
            print(f"{i}. {k} = {v}")

        place2_c1 = builder.c1
        place2_c2_param1 = builder.c2.param1
        assert place2_c1 == place1_c1  # True
        assert place2_c2_param1 == place2_c2_param1  # True

        try:
            builder.c4
        except Exception as e:
            print(e)  # 'Builder' has no attribute 'c4'

        try:
            builder.c2.param6
        except Exception as e:
            print(e)  # 'Mount2' object has no attribute 'param6'

        print(builder)
        assert builder is not None
    test_scope()


def test_builder_config():
    cfg = Config(
        dataset={
            "type": "OCRDatasetV2",
        },
        cfg_type_key="type"
    )

    b = Builder(cfg, name="TrainBuilder", mark=cfg.cfg_type_key)
    print(b.dataset)
    print(COMPONENT_MANAGERS)


if __name__ == '__main__':
    test_builder_config()
