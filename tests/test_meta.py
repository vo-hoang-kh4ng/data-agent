import sys
import os

# Add dgm_agent to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dgm_agent'))
sys.stdout.reconfigure(encoding='utf-8')

from DGM_lambda import EvolutionaryScheduler

s = EvolutionaryScheduler()
s.run_meta_evolution_step(dry_run=True)
