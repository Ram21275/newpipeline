from __future__ import annotations

import json
import unittest
from pathlib import Path


class CubFinalNotebookTests(unittest.TestCase):
    def test_every_code_cell_compiles_and_versions_are_checked_fresh(self) -> None:
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
        self.assertIn("run_stage('post_install_versions'", code)
        self.assertIn("importlib.metadata.version(distribution)", code)
        self.assertIn("version_probe_payload['hf_hub_cache']", code)
        self.assertNotIn("from huggingface_hub.constants import HF_HUB_CACHE\nmodel_revisions", code)
        self.assertNotIn(
            "observed_versions = {name: package_version(name)",
            code,
        )


if __name__ == "__main__":
    unittest.main()
