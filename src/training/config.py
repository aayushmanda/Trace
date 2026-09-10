"""Thin YAML/JSON config loader. CLI kwargs override file keys."""
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    text = path.read_text()
    if path.suffix in {".json"}:
        import json
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        return _simple_yaml(text)


def _simple_yaml(text):
    """Enough YAML for this repo's configs (nested maps, inline lists)."""
    import ast

    def coerce(raw):
        raw = raw.split("#", 1)[0].strip()
        if raw == "":
            return {}
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            if raw in {"true", "True"}:
                return True
            if raw in {"false", "False"}:
                return False
            return raw.strip("'\"")

    root = {}
    stack = [( -1, root)]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = coerce(rest)
        parent[key.strip()] = value
        if value == {}:
            stack.append((indent, parent[key.strip()]))
    return root


def merge_dict(base, override):
    out = dict(base or {})
    for key, value in (override or {}).items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def flatten_experiment(cfg):
    """Lift model/train/data blocks into one namespace for argparse-style code."""
    flat = {}
    for block in ("experiment", "model", "train", "data"):
        if isinstance(cfg.get(block), dict):
            flat.update(cfg[block])
    for key, value in cfg.items():
        if key not in {"experiment", "model", "train", "data"} and not isinstance(value, dict):
            flat[key] = value
    return flat


def load_experiment(path, cli=None):
    cfg = load_yaml(path) if path else {}
    flat = flatten_experiment(cfg)
    if cli:
        for key, value in vars(cli).items() if not isinstance(cli, dict) else cli.items():
            if key in {"config", "action"}:
                continue
            if value is not None:
                flat[key] = value
    return Namespace(**flat), cfg
