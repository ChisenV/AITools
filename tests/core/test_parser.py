# import sys, os
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# AITools_Path = os.path.dirname(BASE_DIR)
# print(BASE_DIR, AITools_Path)
# sys.path.append(AITools_Path)
# sys.path.append(BASE_DIR)

from AITools import XMLParser, JSONParser, YAMLParser


def test_parser():
    print()
    data = XMLParser().load(r"E:\python_project\AIToolsV2\AITools\tests\core\test_config.xml")
    JSONParser().dump(data, r"G:\project\AITools\tests\core\test_config.json")
    YAMLParser().dump(data, r"G:\project\AITools\tests\core\test_config.yaml")
    print(data)

    data1 = JSONParser().load(r"G:\project\AITools\tests\core\test_config.json")
    YAMLParser().dump(data, r"G:\project\AITools\tests\core\test_config_json.yaml")

    data2 = YAMLParser().load(r"G:\project\AITools\tests\core\test_config.yaml")
    print(data1, data2)
