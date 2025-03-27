import copy
import os
from pathlib import Path

import yaml
from ast import literal_eval
from typing import Any, Dict, Optional, List, Type, Union

from .parser import ParserPlugin, SUPPORTED_EXTENSIONS

BASE_KEY = '_base_'
INHERIT_KEY = '_inherit_'
TYPE_KEY = '_type_'


class Config(object):
    """
    Enhanced configuration management class, support:
        - Load from YAML/JSON file or initialize directly via kwargs
        - Multi-file inheritance ('_base_' field)
        - Dynamic parameter coverage ('opts' parameter)
        - Secure deep copy and dictionary operation
    """

    _parsers: Dict[str, Type[ParserPlugin]] = SUPPORTED_EXTENSIONS

    def __init__(
            self,
            path: Optional[str] = None,
            *,
            opts: Optional[list] = None,
            cfg_name: str = None,
            cfg_parser: Optional[Type[ParserPlugin]] = None,
            cfg_base_key: str = BASE_KEY,
            cfg_inherit_key: str = INHERIT_KEY,
            cfg_type_key: str = TYPE_KEY,
            cfg_strict_mode: bool = False,
            **kwargs
    ):
        """
        Args:
            path: Configuration file path (YAML/JSON support)
            opts: Parameter override list (["key=value"])
            cfg_name: Configuration name
            cfg_parser: Custom parser (if not specified, will be determined by file extension)
            cfg_strict_mode: Strict mode (disable unknown configuration items)
            **kwargs: Configuration items passed in directly (precedence over files)
        """
        self._name = cfg_name
        self._parser = cfg_parser
        self.cfg_base_key = cfg_base_key
        self.cfg_inherit_key = cfg_inherit_key
        self.cfg_type_key = cfg_type_key
        self._strict_mode = cfg_strict_mode
        self._cfg = {}

        # Load configuration from file (if any)
        file_config = self._load_config_file(path) if path else {}

        #  Merge all configuration sources (file < base class < kwargs < opts)
        merged = self._merge_configs(file_config, os.path.dirname(path) if path else None)
        merged = self._deep_merge(merged, kwargs)
        self._merged_dict = self._update_with_cli_opts(merged, opts or [])
        self._cfg = copy.deepcopy(self._merged_dict)

    @classmethod
    def register_parser(cls, extension: str, parser: Type[ParserPlugin]):
        """Register a new file format parser"""
        cls._parsers[extension.lower()] = parser

    @classmethod
    def get_parser(cls, extension: str) -> Type[ParserPlugin]:
        """
        Gets the parser for the corresponding suffix
        Args:
            extension: File suffix
        Returns:
            Parser class
        Raises:
            ValueError: Unsupported file format
        """
        ext = extension.lower()
        if ext not in cls._parsers:
            raise ValueError(f"Unsupported config format: {ext}. "
                             f"Supported: {list(cls._parsers.keys())}")
        return cls._parsers[ext]

    @property
    def dic(self) -> Dict[str, Any]:
        """Get a safe copy of the configuration dictionary"""
        return copy.deepcopy(self._cfg)

    def _load_config_file(self, path: str) -> Dict[str, Any]:
        """Load and parse the configuration file"""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in self._parsers:
            raise ValueError(f"Unsupported config format: {suffix}. Supported: {self._parsers.keys()}")

        self._parser = self._parser if self._parser else self.get_parser(path.suffix)
        config = self._parser.load(path)

        # Handle inheritance
        if self.cfg_base_key in config:
            base_dir = path.parent
            base_files = config[self.cfg_base_key]
            base_files = [base_files] if isinstance(base_files, str) else base_files
            for bf in base_files:
                base_path = bf if Path(bf).is_absolute() else base_dir / bf
                base_config = Config(
                    path=str(base_path),
                    strict_mode=self._strict_mode,
                    parser=self._parser
                ).dic
                config = self._deep_merge(base_config, config)

        return config

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    def __setitem__(self, key: str, value: Any):
        self._cfg[key] = value

    def __iter__(self):
        return iter(self._cfg)

    def __contains__(self, key: str) -> bool:
        """Check whether the key is in config"""
        return key in self._cfg

    @property
    def name(self) -> str:
        """The config's name"""
        return self._name

    def items(self):
        """Get all configuration items"""
        return self._cfg.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration item with a default value"""
        return self._cfg.get(key, default)

    def copy(self) -> 'Config':
        """Returns an isolated deep copy"""
        return Config(**copy.deepcopy(self._cfg))

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries"""
        merged = copy.deepcopy(base)

        for key, val in update.items():
            # Key check in strict mode
            if self._strict_mode and base and key not in base:
                raise KeyError(f"Unexpected config key: {key}")

            if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._deep_merge(merged[key], val)
            else:
                merged[key] = copy.deepcopy(val)

        return merged

    def _merge_configs(self, config: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
        """Recursively merge base configs"""
        base_files = config.get(self.cfg_base_key, [])
        if isinstance(base_files, str):
            base_files = [base_files]

        merged = {}
        for bf in base_files:
            base_path = bf if os.path.isabs(bf) else os.path.join(base_dir, bf)
            if not os.path.exists(base_path):
                raise FileNotFoundError(f"Base config not found: {base_path}")

            base_config = Config(base_path, strict_mode=self._strict_mode).dic
            merged = self._deep_merge(base_config, merged)

        return self._deep_merge(merged, config)

    def _update_with_cli_opts(self, config: Dict[str, Any], opts: List[str]) -> Dict[str, Any]:
        """Safely apply CLI overrides"""
        updated = copy.deepcopy(config)

        for opt in opts:
            if '=' not in opt:
                raise ValueError(f"Invalid opts format: {opt}. "
                                 f"--opts params should be key=value, such as "
                                 f"`--opts batch_size=1 test_config.scales=0.75,1.0,1.25`, "
                                 f"but got ({opt})")
            key_part, value = opt.split('=', 1)  # Only divide the first equal sign
            keys = key_part.split('.')

            try:
                value = literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # Preserve string format

            current = updated
            for k in keys[:-1]:
                if k not in current:
                    if self._strict_mode:
                        raise KeyError(f"Invalid config key path: {key_part}")
                    current[k] = {}  # Automatically create nested structures
                current = current[k]

            final_key = keys[-1]
            if self._strict_mode and final_key not in current:
                raise KeyError(f"Invalid config key: {key_part}")

            current[final_key] = value

        return updated

    def __str__(self) -> str:
        return ("+ " + "="*96 + "\n+ " +
                yaml.dump(self._cfg, allow_unicode=True, sort_keys=False)
                .replace("\n", "\n+ ")
                + "="*96)

    def dump(self, path: str, overwrite: bool = True) -> None:
        """Save the current configuration to a file"""
        dump(self, path, self._parser, overwrite)


def dump(
        config: Union[Config, dict],
        path: str,
        parser: Optional[ParserPlugin] = None,
        overwrite: bool = True
) -> None:
    """
    Save the current configuration to a file

    Args:
        config: config objects
        path: Save path (format automatically selected according to suffix)
        parser: Save format parser
        overwrite: Whether to overwrite existing files
    Raises:
        FileExistsError: If the file already exists and `overwrite` is False
        ValueError: If the parser is None when config is a dict object
    """
    file_path = Path(path).resolve()
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File {file_path} already exists")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, Config):
        parser = parser if parser else config.get_parser(file_path.suffix)
        parser.dump(config.dic, file_path)
    else:
        if parser is None:
            raise ValueError("Parser is required when config is a dict object")
        parser.dump(config, file_path)
