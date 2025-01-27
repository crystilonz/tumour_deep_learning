from pathlib import Path
import json


def make_parent(f:Path):
    if not f.parent.exists():
        f.parent.mkdir(parents=True)


def dict_json_pretty(d: dict, f:Path):
    make_parent(f)
    if f.suffix != '.json':
        f = f.with_suffix('.json')
    with open(f, 'w') as target:
        json.dump(d, target, indent=4)