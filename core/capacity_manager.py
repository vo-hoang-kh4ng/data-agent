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

    def allocate_budget_for_dacode(self, task_hardness: str = "Medium", task_category: str = "",
                                      epiplexity_score: float = None) -> dict:
        """Cấp ngân sách cho DA-Code benchmark.

        v2 FIX: Lower temperature across the board for deterministic code generation.
        Previous version pushed temperature to 0.5 for ALL tasks due to flat NCD signal.
        Now uses question complexity score [0.3, 1.5] that actually discriminates.

        Budget strategy:
        - Hardness → base budget (primary)
        - Complexity score → fine-tune retries within band (secondary)
        - Temperature stays LOW (0.15–0.3) for deterministic code
        - Statistical Analysis → +1 retry bonus
        """
        print(f"📊 Đánh giá ngân sách DA-Code (Hardness: {task_hardness}, Category: {task_category}, "
              f"Complexity: {epiplexity_score:.3f})…" if epiplexity_score is not None
              else f"📊 Đánh giá ngân sách DA-Code (Hardness: {task_hardness}, Category: {task_category})…")

        # Phase 1: Base budget từ task hardness (LOW temperature for deterministic code)
        if task_hardness == "Easy":
            budget = {
                "max_retries": 3,
                "num_candidates": 1,
                "temperature": 0.15,
            }
        elif task_hardness == "Hard":
            budget = {
                "max_retries": 5,
                "num_candidates": 1,
                "temperature": 0.3,
            }
        else:  # Medium (default)
            budget = {
                "max_retries": 4,
                "num_candidates": 1,
                "temperature": 0.2,
            }

        # Phase 2: Complexity-driven fine-tuning (question analysis score [0.3, 1.5])
        if epiplexity_score is not None:
            if epiplexity_score < 0.6:
                # Simple task — keep base budget, no changes needed
                print(f"   -> 🟢 Simple task (complexity={epiplexity_score:.3f}): base budget.")
            elif epiplexity_score <= 1.0:
                # Standard complexity — +1 retry for safety
                budget["max_retries"] += 1
                print(f"   -> 🟡 Standard (complexity={epiplexity_score:.3f}): +1 retry → {budget['max_retries']}.")
            else:
                # Complex (statistical tests, multi-table, etc.) — +2 retries
                budget["max_retries"] += 2
                print(f"   -> 🔴 Complex (complexity={epiplexity_score:.3f}): +2 retries → {budget['max_retries']}.")
        else:
            # No complexity score — use hardness-only defaults
            print(f"   -> 📋 Hardness-only budget: {budget['max_retries']} retries, temp={budget['temperature']}.")

        # Bonus retry cho Statistical Analysis (hardest category per results)
        if task_category and "statistical" in task_category.lower():
            budget["max_retries"] += 1
            print(f"   -> 📈 Statistical Analysis bonus: +1 retry → {budget['max_retries']}")

        return budget
