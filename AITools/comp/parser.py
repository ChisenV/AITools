from abc import ABC
from pathlib import Path
from typing import Dict, Any, List, Union

__all__ = ['YAMLParser', 'JSONParser', 'XMLParser']

from AITools.core.manager import ComponentManager

PARSERS = ComponentManager("parsers")


class Parser(ABC):
    """Parser base class"""
    _SUPPORTED_EXTENSIONS = []

    @classmethod
    def parsable_file_extensions(cls) -> List[str]:
        """Enable the parser extension"""
        return cls._SUPPORTED_EXTENSIONS

    @classmethod
    def load(cls, *args, **kwargs) -> Any:
        """Load data from file"""
        raise NotImplementedError

    @classmethod
    def dump(cls, *args, **kwargs) -> None:
        """Dump data to file"""
        raise NotImplementedError


# --------------------- YAML plugin implement ---------------------
@PARSERS.register_component
class YAMLParser(Parser):
    """YAML format plugin"""
    _SUPPORTED_EXTENSIONS = [".yaml", ".yml"]

    @classmethod
    def load(cls, path: Union[str, Path], encoding='utf-8', **kwargs) -> Dict[str, Any]:
        import yaml
        with open(path, 'r', encoding=encoding) as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def dump(cls, data: Dict[str, Any], path: Union[str, Path], encoding='utf-8', **kwargs) -> None:
        import yaml
        with open(path, 'w', encoding=encoding) as f:
            yaml.safe_dump(data, f, allow_unicode=True, indent=kwargs.get('indent', 2))


# --------------------- JSON plugin implement ---------------------
@PARSERS.register_component
class JSONParser(Parser):
    """JSON format plugin"""
    _SUPPORTED_EXTENSIONS = [".json"]

    @classmethod
    def load(cls, path: Union[str, Path], encoding='utf-8', **kwargs) -> Dict[str, Any]:
        import json
        with open(path, 'r', encoding=encoding) as f:
            return json.load(f) or {}

    @classmethod
    def dump(cls, data: Dict[str, Any], path: Union[str, Path], encoding='utf-8', **kwargs) -> None:
        import json
        with open(path, 'w', encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=kwargs.get('indent', 2))


# --------------------- XML plugin implement ---------------------
@PARSERS.register_component
class XMLParser(Parser):
    """XML format plugin (attribute/element/text conversion)"""
    _SUPPORTED_EXTENSIONS = [".xml"]

    @classmethod
    def load(
            cls,
            path: Union[str, Path],
            fmt_att2key: str = '_{}_',
            unlabeled_text_key: str = '#text',
            **kwargs
    ) -> Dict[str, Any]:
        """
        Load the data from a file
        Args:
            path: the path of the file
            fmt_att2key: the format of the key of the attribute on the tag, e.g. '_{}_'
            unlabeled_text_key: the key of the text of the element without label, e.g. '#text'
            **kwargs:

        Returns:
            data: a dictionary
        """
        cls.validate_fmt_only_one_brace(fmt_att2key)

        from xml.etree import ElementTree as ET

        def parse_element(element: ET.Element) -> Dict:
            result = {}
            if element.attrib:
                result.update({fmt_att2key.format(k): v for k, v in element.attrib.items()})
            for child in element:
                child_data = parse_element(child)
                key = child.tag
                if key in result:
                    if not isinstance(result[key], list):
                        result[key] = [result[key]]
                    result[key].append(child_data)
                else:
                    result[key] = child_data
            text = element.text.strip() if element.text else ""
            if text:
                if len(result) > 0:
                    result[unlabeled_text_key] = text
                else:
                    result = cls.auto_convert(text)
            return result

        tree = ET.parse(path)
        root = tree.getroot()
        return {root.tag: parse_element(root)}

    @classmethod
    def dump(
            cls,
            data: Dict[str, Any],
            path: Union[str, Path],
            fmt_att2key: str = '_{}_',
            unlabeled_text_key: str = '#text',
            encoding='utf-8',
            indent='\t',
            **kwargs
    ) -> None:
        """
        Save the data to a file
        Args:
            data: need to dump data
            path: save path
            fmt_att2key: attribute key format, e.g. _{}_
            unlabeled_text_key: unlabeled text key, e.g. #text
            encoding: encoding
            indent: indent
            **kwargs:
        """
        cls.validate_fmt_only_one_brace(fmt_att2key)

        import re
        from xml.etree import ElementTree as ET
        import xml.dom.minidom as minidom
        # Generate attribute key matching patterns(e.g. @{} → ^@(.*?) $)
        prefix, suffix = fmt_att2key.split("{}")
        attr_pattern = re.compile(
            r"^{}(.*?){}$".format(re.escape(prefix), re.escape(suffix))
        )

        def build_element(parent: ET.Element = None, tag: str = '', value: Any = None) -> ET.Element:
            elem = ET.Element(tag) if parent is None else ET.SubElement(parent, tag)
            if isinstance(value, dict):
                # processing attributes
                attrs = {}
                for k, v in value.items():
                    if k == unlabeled_text_key:
                        elem.text = str(value[unlabeled_text_key])
                        continue
                    match = attr_pattern.match(k)
                    if match and isinstance(v, (str, int, float, bool)):
                        attrs[match.group(1)] = str(v)
                        continue
                    # Handle child elements recursively
                    if isinstance(v, list):
                        for it in v:
                            build_element(elem, k, it)
                    else:
                        build_element(elem, k, v)
                elem.attrib.update(attrs)
            else:
                elem.text = str(value)

            return elem

        root_tag = next(iter(data))
        root = build_element(None, root_tag, data[root_tag])
        if indent != '':
            rough_str = ET.tostring(root, encoding=encoding, method="xml")
            dom = minidom.parseString(rough_str)
            pretty_str = dom.toprettyxml(indent=indent, encoding=encoding)
            with open(path, "wb") as f:
                f.write(pretty_str)
        else:
            ET.ElementTree(root).write(path, encoding=encoding, xml_declaration=True)

    @staticmethod
    def auto_convert(text: str) -> Any:
        """Automatic type conversion of text content"""
        text = text.strip()
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text

    @staticmethod
    def validate_fmt_only_one_brace(fmt: str) -> None:
        """
        Verify the validity of the fmt_att2key format string
        request:
            - Must contain and only a pair of consecutive `{}`
            - `{`'` must come before `}`
        """
        # Check the number of '{}' pairs
        if fmt.count("{") != 1 or fmt.count("}") != 1:
            raise ValueError(f"Invalid fmt_att2key: '{fmt}'. Must contain exactly one '{{}}' pair")

        # Check the order and continuity of '{}'
        left_idx = fmt.find("{")
        right_idx = fmt.find("}")

        if left_idx == -1 or right_idx == -1:
            raise ValueError(f"Invalid fmt_att2key: '{fmt}'. Missing '{{' or '}}'")

        if left_idx >= right_idx:
            raise ValueError(f"Invalid fmt_att2key: '{fmt}'. '{{' must come before '}}'")
