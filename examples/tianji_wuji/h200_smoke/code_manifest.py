"""Hash repository sources and one job recipe without traversing job outputs."""

import hashlib
from pathlib import Path


def build_code_manifest(repository: Path, config: Path) -> dict[str, str]:
    repository = repository.resolve()
    files = [*repository.joinpath("src/lerobot").rglob("*.py")]
    files.extend(repository / name for name in ("pyproject.toml", "uv.lock"))
    files.extend(
        path
        for path in repository.joinpath("examples/tianji_wuji").rglob("*")
        if path.suffix in (".py", ".json", ".sh", ".md")
    )
    result = {}
    for path in files:
        resolved = path.resolve()
        if path.is_file() and resolved.is_relative_to(repository):
            with path.open("rb") as stream:
                result[resolved.relative_to(repository).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    # A continuation recipe lives in jobs/<job>, outside the Git checkout.
    # Record that exact file, never recursively inspect its checkpoint directory.
    with config.open("rb") as stream:
        result["job_config/train_config.json"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result
