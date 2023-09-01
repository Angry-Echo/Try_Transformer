"""
注意 json_load 和 json_loads 大相径庭
"""
import json


def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        return json.load(f)


def yield_dialog(path):
    while True:
        with open(path, 'r', encoding='UTF-8') as f:
            for dialog in f:
                yield json.loads(dialog)
