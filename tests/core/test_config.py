from AITools import Config, Builder, logger
from AITools.core.config import dump as config_dump

yaml_file = r"E:\python_project\PaddleSeg-release-2.9\configs\pp_liteseg\wirebonding_fusion_pp_liteseg_stdc1_camvid_1024x864_10k.yml"
xml_file = r"E:\python_ai_dataset\General-SeWen-XiDong-Version3-788Pics\train\Annotations\PinHole@201@201@1@pin0@ID27417(63.44)-NG.xml"


def test_config_init():
    print()

    cfg = Config(yaml_file, opt=
                 ["train_dataset.num_classes=6",
                  "class_type.a=1",
                  "class_type.b=2"],
                 )
    print(cfg)

    for k, v in cfg.items():
        print(k, v)


def test_config_get_copy():
    print()

    cfg = Config(
        name="dfsfsd",
        path=yaml_file,
        opts=["train_dataset.num_classes=6", "class_type.a=1", "class_type.b=2"],
    )
    print(cfg)
    assert str(cfg.get("class_type")) == "{'a': 1, 'b': 2}", "get is not right"
    assert cfg.get("class_type.a") is None
    assert cfg.get("class_type.b") is None
    assert cfg.get("class_type.c") is None
    assert cfg.get("class_type.c", 3) == 3

    cfg_copy = cfg.copy()
    assert cfg_copy != cfg, "copy is not deep copy"
    assert str(cfg_copy) == str(cfg), "copy is not deep copy"


def test_builder_config():
    print()

    cfg = Config(yaml_file)

    ber = Builder(cfg, mark="type")
    print(ber)


def test_config_xml():
    print()

    cfg = Config(xml_file)
    for l in str(cfg).split('\n'):
        logger.log(logger.INFO, f"{l}")

    config_dump(cfg, './test_config.xml')


def test_config_dataclass():
    import copy
    import json

    class DataClassBase:
        """
        Implemented using classes
        In contrast to dictionaries, data classes are refined under the ide.
        """

        def __new__(cls, **kwargs):
            print("new", kwargs, cls.__dict__)
            self = super().__new__(cls)
            self.__dict__ = copy.deepcopy({k: v for k, v in cls.__dict__.items() if not k.startswith('__')})
            return self

        def __init__(self, **kwargs):
            print("init", kwargs)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __call__(self, ) -> dict:
            return self.get_dict()

        def get_dict(self):
            return {k: v.get_dict() if isinstance(v, DataClassBase) else v for k, v in self.__dict__.items()}

        def get_json(self):
            return json.dumps(self.get_dict(), ensure_ascii=False, indent=4)

        def __str__(self):
            return f"{self.__class__}    {self.get_json()}"

        def __getitem__(self, item):
            return getattr(self, item)

        def __setitem__(self, key, value):
            setattr(self, key, value)

    class MailHandlerConfig(DataClassBase):  # replace Config class
        mailhost: tuple = ()
        fromaddr: str = "dfs"
        toaddrs: tuple = ()
        subject: str = 'xx project mail log alarm'
        credentials: tuple = ()
        secure = None
        timeout = 5.0
        is_use_ssl = True
        mail_time_interval = 60

    print(MailHandlerConfig())
    print(MailHandlerConfig(
        mailhost=('localhost', 25),
        fromaddr='dfs',
        toaddrs=('dfs',),
        subject='xx project mail log alarm',
        credentials=(),
        secure=None,
        timeout=5.0,
        is_use_ssl=True,
        mail_time_interval=60))
