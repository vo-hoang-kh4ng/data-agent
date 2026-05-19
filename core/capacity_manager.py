class DynamicCapacityManager:
    """Quản lý ngân sách tính toán thích ứng dựa trên Epiplexity.
    
    - `base_retries` và `base_candidates` là mức tối thiểu.
    - `allocate_budget` trả về dict chứa:
        * max_retries
        * num_candidates
        * model (tên mô hình LLM)
        * temperature (độ ngẫu nhiên)
    """
    def __init__(self, base_retries: int = 1, base_candidates: int = 1):
        self.base_retries = base_retries
        self.base_candidates = base_candidates

    def allocate_budget(self, epiplexity_score: float, current_iteration: int) -> dict:
        """Tính ngân sách dựa trên độ khó (Epiplexity) và vòng lặp.
        
        Parameters
        ----------
        epiplexity_score: float
            Điểm độ phức tạp của task (các giá trị thường nằm trong khoảng 0‑2).
        current_iteration: int
            Số vòng tiến hóa hiện tại, bắt đầu từ 0.
        """
        print(f"📊 Đánh giá ngân sách cấp phát (Epiplexity: {epiplexity_score:.2f})…")
        # 1. Bài toán dễ
        if epiplexity_score < 0.5:
            print("   -> 🟢 Bài toán dễ: Dùng ngân sách tối thiểu.")
            return {
                "max_retries": self.base_retries,
                "num_candidates": 1,
                "model": "llama-3.1-8b-instant",
                "temperature": 0.2,
                "current_iteration": current_iteration
            }
        # 2. Vùng Goldilocks
        elif 0.5 <= epiplexity_score <= 1.2:
            bonus_retries = min(current_iteration // 10, 3)
            print(f"   -> 🟡 Vùng học tập lý tưởng: Cấp thêm {bonus_retries} retries.")
            return {
                "max_retries": self.base_retries + bonus_retries,
                "num_candidates": self.base_candidates + 1,
                "model": "llama-3.1-8b-instant",
                "temperature": 0.6,
                "current_iteration": current_iteration
            }
        # 3. Bài toán cực khó
        else:
            print("   -> 🔴 Bài toán siêu khó: Kích hoạt Tree-of-Thought, bung ngân sách tối đa!")
            return {
                "max_retries": self.base_retries + 4,
                "num_candidates": 3,
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.8,
                "current_iteration": current_iteration
            }
