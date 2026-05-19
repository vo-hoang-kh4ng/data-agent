import json
import os

class PolyglotBenchmark:
    def __init__(self):
        # 10 bài test cơ bản để kiểm tra khả năng lập trình nền tảng
        self.initial_10_tasks = [
            "go__dominoes", "cpp__all-your-base", "python__dominoes", 
            "java__sgf-parsing", "javascript__robot-name", "rust__variable-length-quantity", 
            "python__beer-song", "go__book-store", "javascript__bottle-song", "rust__bowling"
        ]
        
        # 50 bài test mở rộng (chỉ chạy nếu qua được vòng 1)
        self.extended_50_tasks = [
            "javascript__queen-attack", "rust__wordy", "python__dot-dsl",
            "java__satellite", "cpp__diamond", "rust__accumulate",
            "go__error-handling", "cpp__queen-attack", "rust__poker", "python__sgf-parsing",
            # Thêm các bài test mở rộng giả lập cho đủ số lượng đánh giá
            "rust__grade-school", "python__grade-school", "javascript__grade-school",
            "go__grade-school", "cpp__grade-school", "java__grade-school",
            "rust__allergies", "python__allergies", "javascript__allergies",
            "go__allergies", "cpp__allergies", "java__allergies",
            "rust__anagram", "python__anagram", "javascript__anagram",
            "go__anagram", "cpp__anagram", "java__anagram",
            "rust__binary-search", "python__binary-search", "javascript__binary-search",
            "go__binary-search", "cpp__binary-search", "java__binary-search",
            "rust__bob", "python__bob", "javascript__bob",
            "go__bob", "cpp__bob", "java__bob",
            "rust__clock", "python__clock", "javascript__clock",
            "go__clock", "cpp__clock", "java__clock",
            "rust__collatz-conjecture", "python__collatz-conjecture", "javascript__collatz-conjecture",
            "go__collatz-conjecture", "cpp__collatz-conjecture", "java__collatz-conjecture"
        ]

    def load_task(self, task_id):
        """Mô phỏng việc kéo đề bài và Unit Test ẩn từ HuggingFace/Github"""
        print(f"📥 Đang tải bài toán đa ngôn ngữ: {task_id}...")
        parts = task_id.split("__")
        lang = parts[0]
        name = parts[1] if len(parts) > 1 else task_id
        return {
            "task_id": task_id,
            "language": lang,
            "description": f"Solve the {name} problem in {lang} from scratch.",
            "hidden_test": f"assert solve() == expected_output" # Test case ẩn để đo pass@1
        }
