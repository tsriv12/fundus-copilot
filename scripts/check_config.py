from pathlib import Path
import yaml

base_path = Path("configs/base.yaml")
paths_path = Path("configs/paths.yaml")

with open(base_path, "r") as f:
    base_cfg = yaml.safe_load(f)

with open(paths_path, "r") as f:
    paths_cfg = yaml.safe_load(f)

print("Loaded base config:")
print(base_cfg)
print()
print("Loaded paths config:")
print(paths_cfg)
