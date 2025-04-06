import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AITools_Path = os.path.dirname(BASE_DIR)
print(BASE_DIR, AITools_Path)
sys.path.append(AITools_Path)
sys.path.append(BASE_DIR)

from AITools.comp import parser


def test_parser():
    data = parser.XMLParser().load(r"G:\project\AITools\tests\core\test_config.xml")
    parser.JSONParser().dump(data, r"G:\project\AITools\tests\core\test_config.json")
    parser.YAMLParser().dump(data, r"G:\project\AITools\tests\core\test_config.yaml")
    print(data)

    data1 = parser.JSONParser().load(r"G:\project\AITools\tests\core\test_config.json")
    parser.YAMLParser().dump(data, r"G:\project\AITools\tests\core\test_config_json.yaml")

    data2 = parser.YAMLParser().load(r"G:\project\AITools\tests\core\test_config.yaml")
    print(data1, data2)
