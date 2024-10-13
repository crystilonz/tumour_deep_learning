import json
from pathlib import Path
from typing import Any

def save_to_json(data: dict[str, Any],
                 save_path: Path):
    if save_path.suffix != '.json':
        save_path = save_path.parent / (save_path.name + '.json')
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
