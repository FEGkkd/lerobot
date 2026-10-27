"""Route locked CPython 3.12 Linux x86_64 wheels to their original package sources."""

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    special = {"torch", "torchvision", "torchcodec"}
    lock_bytes = args.lock.read_bytes()
    lock = tomllib.loads(lock_bytes.decode())
    blocks = re.split(r"(?m)(?=^[A-Za-z0-9][A-Za-z0-9_.-]*(?:==| @ ))", args.requirements.read_text())
    ordinary = [block for block in blocks if not any(block.startswith(name + "==") for name in special)]
    wheels = {}
    for package in lock["package"]:
        name = package["name"]
        if name not in special or "pytorch.org" not in package.get("source", {}).get("registry", ""):
            continue
        matches = [
            wheel
            for wheel in package["wheels"]
            if "-cp312-cp312-" in unquote(wheel["url"]) and wheel["url"].endswith("x86_64.whl")
        ]
        if len(matches) != 1 or name in wheels:
            raise ValueError(f"Ambiguous CPython 3.12 Linux wheel: {name}")
        wheels[name] = matches[0]
    if set(wheels) != special:
        raise ValueError("The lock must contain exactly the expected three PyTorch-source wheels.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "requirements.pypi.lock.txt").write_text("".join(ordinary))
    (args.output_dir / "requirements.pytorch.lock.txt").write_text(
        "".join(f"{name} @ {wheel['url']} --hash={wheel['hash']}\n" for name, wheel in sorted(wheels.items()))
    )
    (args.output_dir / "environment-sources.json").write_text(
        json.dumps(
            {
                "uv_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
                "target": "cp312-linux-x86_64",
                "pytorch_wheels": wheels,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "Locked PyPI hashes retained; PyTorch, torchvision and TorchCodec use exact original wheel URLs and hashes."
    )


if __name__ == "__main__":
    main()
