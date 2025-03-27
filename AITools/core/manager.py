import inspect

import warnings
from types import MappingProxyType
from typing import Callable, Dict, List, Optional, Tuple, Union


class ComponentManager:
    """
    Implement a manager class to add the new component properly.
    The component can be added as either class or function type.

    Args:
        name (str): The name of component.
        comp_name_getter (Callable): The function to get the name of component.

    Returns:
        A callable object of ComponentManager.

    Examples 1:

        from AITools.core.manager import ComponentManager

        model_manager = ComponentManager()

        class AlexNet: ...
        class ResNet: ...

        model_manager.register_component(AlexNet)
        model_manager.register_component(ResNet)

        # Or pass a sequence alliteratively:
        model_manager.register_component([AlexNet, ResNet])
        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}

    Examples 2:

        # Or an easier way, using it as a Python decorator, while just add it above the class declaration.
        from AITools.core.manager import ComponentManager

        model_manager = ComponentManager()

        @model_manager.register_component
        class AlexNet: ...

        @model_manager.register_component
        class ResNet: ...

        print(model_manager.components_dict)
        # {'AlexNet': <class '__main__.AlexNet'>, 'ResNet': <class '__main__.ResNet'>}
    """

    def __init__(
            self,
            name: Optional[str] = None,
            comp_name_getter: Optional[Callable[[Union[type, Callable]], str]] = None
    ):
        self._components_dict: Dict[str, Union[type, Callable]] = {}
        self._name = name
        self._name_getter = comp_name_getter or (lambda x: x.__name__)

    def __len__(self) -> int:
        return len(self._components_dict)

    def __repr__(self) -> str:
        name_str = self._name if self._name else self.__class__.__name__
        return "{}:{}".format(name_str, list(self._components_dict.keys()))

    def __getitem__(self, item: str) -> Union[type, Callable]:
        if item not in self._components_dict.keys():
            available = list(self._components_dict.keys())
            raise KeyError(
                f"'{item}' does not exist in {self._name or self.__class__.__name__}. Available: {available}")
        return self._components_dict[item]

    @property
    def components_dict(self) -> MappingProxyType:
        return MappingProxyType(self._components_dict)

    @property
    def name(self) -> Optional[str]:
        return self._name

    def _add_single_component(
            self,
            component: Union[type, Callable],
            allow_overwrite: bool = False
    ):
        """
        Add a single component into the corresponding manager.

        Args:
            component (function|class): A new component.
            allow_overwrite (bool): Whether to allow overwrite the existing component. Default: False.

        Raises:
            TypeError: When `component` is neither class nor function.
            KeyError: When `component` was added already.
        """

        # Currently only support class or function type
        if not (inspect.isclass(component) or inspect.isfunction(component)):
            raise TypeError("Expect class/function type, but received {}".
                            format(type(component)))

        # Obtain the internal name of the component
        component_name = self._name_getter(component)

        # Check whether the component was added already
        if component_name in self._components_dict.keys():
            if not allow_overwrite:
                raise KeyError(f"Component '{component_name}' exists. Use allow_overwrite=True to replace.")
            warnings.warn(f"Replacing '{component_name}' with {component}.", UserWarning)

        # Take the internal name of the component as its key
        self._components_dict[component_name] = component

    def register_component(
            self,
            components: Union[type, Callable, List[Union[type, Callable]], Tuple[Union[type, Callable], ...]] = None,
            allow_overwrite: bool = False
    ):
        """
        Add component(s) into the corresponding manager.

        Args:
            components (function|class|list|tuple): Support four types of components.
                1. None: Use it as a decorator.
                2. function: Add a single function.
                3. class: Add a single class.
                4. list/tuple: Add multiple components.
            allow_overwrite (bool): Whether allow overwriting the existing component.

        Returns:
            components (function|class|list|tuple): Same with input components.
        """

        # Check whether the type is a sequence
        if components is None:
            # Allow @manager.register_component(allow_overwrite=True) syntax
            def decorator(comp):
                self._add_single_component(comp, allow_overwrite)
                return comp

            return decorator
        else:
            # Handle direct component(s) addition
            if isinstance(components, (list, tuple)):
                for comp in components:
                    self._add_single_component(comp, allow_overwrite)
            else:
                self._add_single_component(components, allow_overwrite)
            return components


ADAPTERS = ComponentManager("adapters")
DATASETS = ComponentManager("datasets")
DATALOADERS = ComponentManager("dataloaders")
EVALUATORS = ComponentManager("evaluators")
METRICS = ComponentManager("metrics")
TRANSFORMS = ComponentManager("transforms")
CONVERTS = ComponentManager("converts")
PARSER = ComponentManager("parser")

COMPONENT_MANAGERS = [ADAPTERS, DATASETS, DATALOADERS, EVALUATORS, METRICS, TRANSFORMS, CONVERTS, PARSER]
