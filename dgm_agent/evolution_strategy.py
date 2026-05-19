import random
import math
class EvolutionStrategy:
    STRATEGY_NAME = 'novelty_ucb_v2'
    STRATEGY_VERSION = 2
    def select_parent(self, archive: list) -> dict:
        if not archive: return {}
        total_visits = sum(n.get('children_count', 0) for n in archive) + 1
        best_score = -1
        best_node = archive[0]
        for node in archive:
            score = node.get('score', 0.5)
            visits = node.get('children_count', 0)
            ucb = score + 0.1 * math.sqrt(math.log(total_visits) / (visits + 1))
            if ucb > best_score:
                best_score = ucb
                best_node = node
        best_node['children_count'] = best_node.get('children_count', 0) + 1
        return best_node
    def describe(self) -> str:
        return f'Strategy: {self.STRATEGY_NAME} v{self.STRATEGY_VERSION}'
    def fitness(self, archive: list) -> float:
        return 1.0