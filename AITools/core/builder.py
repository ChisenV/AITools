import copy
from typing import Optional, Dict, Any, Union, Callable

from .config import Config, TYPE_KEY
from .logger import get_logger
from .manager import COMPONENT_MANAGERS

logger = get_logger(__name__)


class Builder(object):
    """
    Lazy component builder with support for nested configurations.
    Attributes are dynamically created based on the top-level keys in the config.
    """

    def __init__(
            self,
            config: Union[Config, Dict[str, Any]] = None,
            *,
            components: Optional[list] = None,
            name: str = None,
            mark: str = TYPE_KEY,
            post_build_hooks: Optional[list[Callable[[Any, Dict], None]]] = None
    ):
        """
        Initialize the builder with a configuration dictionary and a list of component classes.
        Args:
            config: (Config, Dict[str, Any]) Configuration dictionary or Config object.
            components: (Optional[list]) List of component classes.
            name: (str) Name of the builder.
            mark: (str) Mark used to identify component configurations in the config dictionary.
            post_build_hooks: (Optional[list[Callable[[Any, Dict], None]]]) List of post-build hooks.
        """
        super().__init__()
        self.config = copy.deepcopy(config) if isinstance(config, dict) else config.copy()
        self.components = components or COMPONENT_MANAGERS or []

        self._name = name
        self._mark = mark
        if isinstance(config, Config) and config.cfg_type_key != mark:
            logger.warning(f'The configuration type key "{config.cfg_type_key}" does not match '
                           f'the builder tag "{mark}". If you are sure you want to use it this '
                           f'way, ignore this warning.')
        self._post_build_hooks = post_build_hooks or []
        # Store the configuration of all top-level components {attr_name: config}
        self._component_configs = {
            k: v for k, v in self.config.items()
            if self._is_component_config(v)
        }
        # Cache created component instances {attr_name: instance}
        self._component_instances = {}

    def __getattr__(self, name: str):
        """Lazily build component when accessing attributes"""
        if name in self._component_instances:
            return self._component_instances[name]

        if name in self._component_configs:
            cfg = copy.deepcopy(self._component_configs[name])
            instance = self._build_component(cfg, seen=set())
            self._component_instances[name] = instance
            return instance

        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def _build_component(self, cfg: Dict, seen: Optional[set] = None) -> Any:
        """Recursive component builder"""
        if not self._is_component_config(cfg):
            return cfg

        seen = seen or set()  # Trace build path
        cfg_id = id(cfg)
        if cfg_id in seen:
            raise RuntimeError("Circular dependency detected in component config")
        seen.add(cfg_id)

        component_cls = None
        try:
            component_type = cfg.pop(self._mark)

            params = {}
            for key, val in cfg.items():
                if self._is_component_config(val):  # Build subcomponents recursively
                    params[key] = self._build_component(val, seen)
                elif isinstance(val, list):
                    params[key] = [
                        self._build_component(item, seen) if self._is_component_config(item) else item for item in val
                    ]
                else:
                    params[key] = val

            component_cls = self._find_component_class(component_type)
            instance = component_cls(**params)
            return self.post_build_hook(instance, cfg)
        except Exception as e:
            if hasattr(component_cls, '__name__'):
                com_name = component_cls.__name__
            else:
                com_name = ''
            raise RuntimeError(
                f"Tried to create a {com_name} object, but the operation has failed. "
                "Please double check the arguments used to create the object.\n"
                f"The error message is: \n{str(e)}")
        finally:
            seen.remove(cfg_id)

    def _is_component_config(self, cfg: Any) -> bool:
        """Check if a value is a component config"""
        return isinstance(cfg, dict) and self._mark in cfg

    def _find_component_class(self, component_type: str) -> type:
        """According to the component type, find the corresponding component class"""
        for manager in self.components:
            if hasattr(manager, "components_dict") and component_type in manager.components_dict:
                return manager.components_dict[component_type]
        raise ValueError(f"Component type '{component_type}' not found")

    def post_build_hook(self, instance, cfg) -> Any:
        """
        Post build hook
        Args:
            instance: Component instance
            cfg: Component config
        Returns:
            Component instance
        """
        for hook in self._post_build_hooks:
            if isinstance(hook, Callable):
                hook(instance, cfg)
        return instance

    @property
    def name(self) -> str:
        """The builder's name"""
        return self._name

    def __str__(self):
        return (f"Builder(name={self._name}):\n    " +
                "\n    ".join(self._component_configs))
