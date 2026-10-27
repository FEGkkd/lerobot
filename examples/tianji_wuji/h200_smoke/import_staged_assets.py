"""Import a checksummed local staging archive into the mounted shared disk."""

import argparse
import fcntl
import json
import tarfile
import time
from pathlib import Path

from prepare_splits import DATASET_REVISION, MODEL_REVISIONS, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bytes", type=int, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--wait-minutes", type=int, default=0)
    args = parser.parse_args()
    work = args.work_root.resolve()
    archive = args.archive.resolve()
    if not str(work).startswith("/inspire/") or not archive.is_relative_to(work):
        raise ValueError("Import must stay in the selected mounted work directory")
    deadline = time.monotonic() + args.wait_minutes * 60
    while not archive.is_file() or archive.stat().st_size != args.bytes:
        if time.monotonic() >= deadline:
            raise RuntimeError("The archive upload is incomplete")
        time.sleep(10)
    if sha256_file(archive) != args.sha256:
        raise ValueError("Transfer checksum mismatch")
    preparation = work / "preparation"
    preparation.mkdir(parents=True, exist_ok=True)
    with (preparation / "asset-import.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                target = (work / member.name).resolve()
                if not target.is_relative_to(work) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            source.extractall(work, filter="data")
        print(json.dumps({"imported": archive.name, "sha256": args.sha256}), flush=True)
        public = preparation / "staged-public-manifest.json"
        cosmos = preparation / "staged-cosmos-manifest.json"
        if not public.is_file() or not cosmos.is_file():
            print("Waiting for the other pinned asset archive; assets.json was not created.", flush=True)
            return
        repositories = json.loads(public.read_text())["repositories"] + [json.loads(cosmos.read_text())]
        expected = {"fegkkd/bag_v3": DATASET_REVISION, **MODEL_REVISIONS}
        if {entry["repo"]: entry["revision"] for entry in repositories} != expected:
            raise ValueError("Staged repository revisions do not match the reviewed job")
        assets = {
            "dataset_root": str(work / "datasets/bag_v3"),
            "dataset_revision": DATASET_REVISION,
            "hf_home": str(work / "hf"),
            "models": {},
        }
        for entry in repositories:
            folder = (work / entry["destination"]).resolve()
            if not folder.is_relative_to(work):
                raise ValueError("Manifest destination escaped the work directory")
            for file in entry["files"]:
                target = (folder / file["path"]).resolve()
                if not target.is_relative_to(folder) or target.stat().st_size != file["size"]:
                    raise ValueError(f"Invalid staged file: {file['path']}")
                if sha256_file(target) != file["sha256"]:
                    raise ValueError(f"Staged file checksum mismatch: {file['path']}")
            if entry["repo"] in MODEL_REVISIONS:
                ref = folder.parent.parent / "refs/main"
                if ref.read_text().strip() != entry["revision"]:
                    raise ValueError("Offline model ref does not match the pinned revision")
                assets["models"][entry["repo"]] = {"revision": entry["revision"], "path": str(folder)}
        pending = work / "assets.import-pending.json"
        pending.write_text(json.dumps(assets, indent=2) + "\n")
        pending.replace(work / "assets.json")
        print(
            json.dumps({"status": "all_shared_assets_verified", "assets": str(work / "assets.json")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
