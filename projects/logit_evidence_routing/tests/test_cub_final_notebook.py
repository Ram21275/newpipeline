from __future__ import annotations

import json
import unittest
from pathlib import Path


class CubFinalNotebookTests(unittest.TestCase):
    def test_every_code_cell_compiles_and_runtime_imports_are_checked(self) -> None:
        path = Path(__file__).resolve().parents[1] / "notebooks" / "CUB_Final_Kaggle.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
        requirements = (
            Path(__file__).resolve().parents[1] / "requirements-cub-final-kaggle.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("huggingface-hub==0.36.2", requirements)
        self.assertIn("run_stage('runtime_import_probe'", code)
        self.assertIn("import transformers", code)
        self.assertNotIn("importlib.metadata.version(distribution)", code)
        self.assertIn("CUB_FINAL_VERSION_JSON=", code)
        self.assertIn("version_probe_payload['hf_hub_cache']", code)
        self.assertNotIn("from huggingface_hub.constants import HF_HUB_CACHE\nmodel_revisions", code)
        self.assertNotIn(
            "observed_versions = {name: package_version(name)",
            code,
        )


if __name__ == "__main__":
    unittest.main()
