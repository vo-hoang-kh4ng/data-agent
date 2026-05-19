import random
import math

class EvolutionStrategy:
    """
    [IMMUTABLE BASELINE] Chiến lược chọn cha mẹ cho vòng tiến hóa DGM.
    File này LÀ PARACHUTE (dự phòng) KHÔNG ĐƯỢC PHÉP THAY ĐỔI.
    Được dùng để rollback khi chiến lược Meta-Evolved liên tục thất bại.
    """
    STRATEGY_NAME = "score_child_prop_v1"
    STRATEGY_VERSION = 1
    
    def select_parent(self, archive: list) -> dict:
        """
        Chọn thế hệ cha mẹ tốt nhất từ Archive để clone + đột biến.
        Baseline: Chọn theo tỉ lệ thuận với điểm số và tỉ lệ nghịch với số lượng con.
        """
        if not archive:
            return {}
            
        if len(archive) == 1:
            return archive[0]
            
        scores = [node.get('score', 0.5) for node in archive]
        children_counts = [node.get('children_count', 0) for node in archive]
        
        weights = []
        for score, children in zip(scores, children_counts):
            s_weight = 1 / (1 + math.exp(-10 * (score - 0.5)))
            c_weight = 1 / (1 + children)
            weights.append(s_weight * c_weight)
            
        total_weight = sum(weights)
        if total_weight == 0:
            probabilities = [1.0 / len(archive)] * len(archive)
        else:
            probabilities = [w / total_weight for w in weights]
            
        chosen = random.choices(archive, weights=probabilities, k=1)[0]
        chosen['children_count'] = chosen.get('children_count', 0) + 1
        
        return chosen
        
    def describe(self) -> str:
        return f"Strategy: {self.STRATEGY_NAME} v{self.STRATEGY_VERSION} (BASELINE)"
        
    def fitness(self, archive: list) -> float:
        if not archive:
            return 0.0
        scores = [node.get('score', 0.0) for node in archive]
        return sum(scores) / len(scores)
