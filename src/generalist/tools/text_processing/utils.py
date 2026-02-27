import yaml
from pathlib import Path

def parse_config(tool_function: str, param: str) -> dict:
    """
    TODO: create error handling?
    """
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if cfg_path.exists():
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        conf_func = cfg.get(tool_function, None)
        if conf_func:
            conf_param = conf_func.get(param, None)

            return conf_param

    return None


def read_local_file(filepath: str):
    if not filepath.startswith("http") or filepath.startswith("www"):
        with open(filepath, "rt") as f:
            content = f.read()
    else:
        raise ValueError(f"Cannot read from non-local resource {filepath}")

    return content