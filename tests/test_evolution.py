import sys, os
sys.stdout.reconfigure(encoding='utf-8')
# Ensure repo root is on path
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "dgm_agent"))

from dgm_agent.DGM_lambda import EvolutionaryScheduler

if __name__ == '__main__':
    # Run a single evolution cycle with meta‑evolution triggered each cycle
    scheduler = EvolutionaryScheduler(meta_evolution_freq=1)
    archive = scheduler.run_evolution_cycle(iterations=1)
    print('=== Archive after run ===')
    for entry in archive:
        print(entry)
