"""Regression coverage for continuation recipes outside the repository."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("code_manifest", Path(__file__).parents[1] / "code_manifest.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CodeManifestTest(unittest.TestCase):
    def test_external_recipe_does_not_traverse_checkpoint_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "code/lerobot"
            for name in (
                "src/lerobot/model.py",
                "pyproject.toml",
                "uv.lock",
                "examples/tianji_wuji/h200_continue/run_h200.sh",
            ):
                path = repository / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source")
            job = root / "jobs/continuation"
            job.mkdir(parents=True)
            config = job / "train_config.json"
            config.write_text('{"steps":300000}')
            (job / "checkpoint.json").write_text("must not enter manifest")
            manifest = module.build_code_manifest(repository, config)
            self.assertEqual(len(manifest), 5)
            self.assertEqual(
                manifest["job_config/train_config.json"], hashlib.sha256(config.read_bytes()).hexdigest()
            )
            self.assertIn("examples/tianji_wuji/h200_continue/run_h200.sh", manifest)
            self.assertFalse(any("checkpoint" in key for key in manifest))
            config.write_text('{"steps":35000}')
            self.assertNotEqual(
                manifest["job_config/train_config.json"],
                module.build_code_manifest(repository, config)["job_config/train_config.json"],
            )

    def test_repository_recipe_is_also_recorded_as_materialized_config(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            (repository / "pyproject.toml").write_text("project")
            (repository / "uv.lock").write_text("lock")
            config = repository / "examples/tianji_wuji/h200_formal/train_config.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}")
            manifest = module.build_code_manifest(repository, config)
            self.assertEqual(
                manifest["job_config/train_config.json"],
                manifest["examples/tianji_wuji/h200_formal/train_config.json"],
            )


if __name__ == "__main__":
    unittest.main()
