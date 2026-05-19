import sys
import os

tests_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(tests_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "dgm_agent"))
sys.stdout.reconfigure(encoding='utf-8')

from DGM_lambda import EvolutionaryScheduler

s = EvolutionaryScheduler()
s.run_meta_evolution_step(dry_run=True)
