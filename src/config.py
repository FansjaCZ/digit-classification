import tomllib
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "config.toml"
with CONFIG_FILE.open("rb") as f:
    data = tomllib.load(f)