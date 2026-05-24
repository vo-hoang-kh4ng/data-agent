import random
import math
import json

class EvolutionStrategy:
    STRATEGY_NAME = 'novelty_adaptive_thompson_sampling_v5'
    STRATEGY_VERSION = 5
    def select_parent(self, archive: list) -> dict:
        if not archive: return {}
        total_visits = sum(n.get('children_count', 0) for n in archive) + 1
        
        # Use a softmax-based strategy to balance exploration and exploitation
        weights = []
        for node in archive:
            score = node.get('score', 0.5)
            visits = node.get('children_count', 0)
            ucb = score + 0.2 * math.sqrt(math.log(total_visits) / (visits + 1))
            novelty = 1 - node.get('novelty_score', 0.5)  # Assuming novelty score ranges from 0 to 1
            temperature = 0.1  # Exploration temperature
            
            # Use a weighted sum of UCB and novelty score with a bonus for high novelty
            weighted_score = ucb + novelty
            weights.append(math.exp(weighted_score / temperature))
        
        # Normalize weights to form a probability distribution
        weights = [w / sum(weights) for w in weights]
        
        # Use the softmax-based weights to select the parent
        best_node = random.choices(archive, weights=weights)[0]
        
        # Implement a bonus for nodes with high novelty scores
        bonus = 0.2
        best_novelty_node = max(archive, key=lambda x: x.get('novelty_score', 0.5))
        best_node['score'] += bonus * (1 - best_novelty_node.get('novelty_score', 0.5))
        
        # Update children count and novelty score
        best_node['children_count'] = best_node.get('children_count', 0) + 1
        best_node['novelty_score'] = best_node.get('novelty_score', 0.0) + 1 / (total_visits + 1)
        best_node['novelty_score'] = max(0.0, min(1.0, best_node['novelty_score']))  # Clip novelty score to [0, 1]
        
        return best_node
    def describe(self) -> str:
        return f'Strategy: {self.STRATEGY_NAME} v{self.STRATEGY_VERSION}'
    def fitness(self, archive: list) -> float:
        return 1.0