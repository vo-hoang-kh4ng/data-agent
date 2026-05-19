import sys, os
sys.stdout.reconfigure(encoding='utf-8')
tests_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(tests_dir)
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "dgm_agent"))

from dgm_agent.DGM_lambda import EvolutionaryScheduler

if __name__ == '__main__':
    # Run a single evolution cycle with meta‑evolution triggered each cycle
    scheduler = EvolutionaryScheduler(meta_evolution_freq=1)
    archive = scheduler.run_evolution_cycle(iterations=1)
    print('=== Archive after run ===')
    for entry in archive:
        print(entry)
