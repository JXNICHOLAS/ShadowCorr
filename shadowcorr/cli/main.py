"""
Single entry: ``shadowcorr <train|eval|preprocess> [hydra overrides...]``

Forwards to the same Hydra apps as ``python -m shadowcorr.{train_app,eval_app,preprocess_app}``.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "Usage: shadowcorr <train|eval|preprocess> [Hydra overrides...]\n"
            "Examples:\n"
            "  shadowcorr train data.train_dir=/data/train data.valid_dir=/data/valid\n"
            "  shadowcorr eval input_dir=/data/test model_path=/path/to/checkpoint.pth\n"
            "  shadowcorr preprocess\n"
            "\n"
            "Or call modules directly:\n"
            "  python -m shadowcorr.train_app --help\n"
        )
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    cmd = sys.argv[1].lower()
    rest = sys.argv[2:]
    exe = sys.executable
    env = os.environ.copy()
    # Editable install: ensure repository root is on PYTHONPATH when ``setup.py`` sits next to ``shadowcorr/``.
    import shadowcorr as _sc

    pkg_dir = os.path.dirname(os.path.abspath(_sc.__file__))
    repo_root = os.path.dirname(pkg_dir)
    if os.path.isfile(os.path.join(repo_root, "pyproject.toml")):
        py_path = env.get("PYTHONPATH", "")
        sep = os.pathsep
        env["PYTHONPATH"] = f"{repo_root}{sep}{py_path}" if py_path else repo_root

    modules = {
        "train": "shadowcorr.train_app",
        "eval": "shadowcorr.eval_app",
        "preprocess": "shadowcorr.preprocess_app",
    }
    if cmd not in modules:
        print(f"Unknown command: {cmd!r}. Use train, eval, or preprocess.", file=sys.stderr)
        sys.exit(1)

    mod = modules[cmd]
    rc = subprocess.call([exe, "-m", mod] + rest, env=env)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
