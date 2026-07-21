import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_integration_hook.py"
SPEC = importlib.util.spec_from_file_location("install_integration_hook", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InstallHookTests(unittest.TestCase):
    def test_both_hooks_patch_and_remain_valid_python(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            package = project / "dgm_agent"
            package.mkdir()
            orchestrator = package / "dacode_orchestrator.py"
            orchestrator.write_text(
                "def run():\n"
                "    max_retries = 5\n"
                "    print(f'  Budget: {max_retries} retries')\n",
                encoding="utf-8",
            )
            blackboard = package / "blackboard.py"
            blackboard.write_text(
                "import os\n"
                "from pathlib import Path\n"
                "class FileAgent:\n"
                "    def __init__(self, **kwargs): pass\n"
                "def build_file_agents(data_lake_dir):\n"
                "    data_root = Path(data_lake_dir)\n"
                "    all_files = ['x']\n"
                "    clustering_method = os.environ.get('DACODE_CLUSTERING_METHOD', 'kmeans').lower()\n"
                "    return []\n",
                encoding="utf-8",
            )
            MODULE.patch_reflexion(project)
            MODULE.patch_frozen_clustering(project)
            orchestrator_source = orchestrator.read_text(encoding="utf-8")
            blackboard_source = blackboard.read_text(encoding="utf-8")
            self.assertIn(MODULE.REFLEXION_MARKER, orchestrator_source)
            self.assertIn(MODULE.CLUSTER_MARKER, blackboard_source)
            ast.parse(orchestrator_source)
            ast.parse(blackboard_source)
            self.assertTrue(orchestrator.with_suffix(".py.pre_evolution.bak").exists())
            self.assertTrue(blackboard.with_suffix(".py.pre_evolution.bak").exists())


if __name__ == "__main__":
    unittest.main()
