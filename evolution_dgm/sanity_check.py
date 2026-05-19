import ast
import subprocess
import sys
import os

# Configure terminal output to support UTF-8 characters on Windows/Linux
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def is_valid_agent(new_agent_code_path: str) -> bool:
    """
    Giai đoạn 1: Sanity Check để đảm bảo Agent mới sinh ra không bị "mất trí nhớ" 
    hoặc hỏng chức năng chỉnh sửa code cơ bản.
    
    Lấy cảm hứng trực tiếp từ thuật toán gác cổng của Darwin Gödel Machine (DGM).
    """
    print(f"[KIỂM TRA] Đang kiểm tra sinh tồn cho phiên bản: {new_agent_code_path}")

    # ---------------------------------------------------------
    # BƯỚC 1: KIỂM TRA BIÊN DỊCH TĨNH (COMPILE CHECK)
    # Đảm bảo LLM không viết ra code sai cú pháp (thiếu dấu hai chấm, sai thụt lề...)
    # ---------------------------------------------------------
    try:
        with open(new_agent_code_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
            ast.parse(code_content)  # Phân tích cú pháp tĩnh
        print("[OK] [BƯỚC 1] Cú pháp hợp lệ (AST syntax OK)!")
    except SyntaxError as e:
        print(f"[FAIL] [BỊ LOẠI] Lỗi cú pháp Python. Agent tự viết code lỗi: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] [BỊ LOẠI] Không thể đọc tệp mã nguồn để kiểm tra: {e}")
        return False

    # ---------------------------------------------------------
    # BƯỚC 2: KIỂM TRA CHỨC NĂNG CỐT LÕI (BASIC FUNCTIONALITY)
    # Ép Agent chạy qua 10 bài test siêu cơ bản (như Polyglot/SWE-bench subset)
    # ---------------------------------------------------------
    # Lưu ý: Cờ --maxfail=1 giúp dừng ngay lập tức ở lỗi đầu tiên để tiết kiệm tiền API
    sanity_test_cmd = [
        sys.executable,  # Use current python executable environment
        "-m", "pytest", 
        "tests/sanity_10_tasks.py", 
        "--maxfail=1", 
        "--disable-warnings",
        "-q"
    ]
    
    try:
        # Sử dụng timeout=60 để bảo vệ hệ thống khỏi treo máy nhưng đủ thời gian chạy Jupyter Kernels trên Windows
        result = subprocess.run(
            sanity_test_cmd, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        
        if result.returncode != 0:
            print("[FAIL] [BỊ LOẠI] Mất chức năng cốt lõi. Agent không vượt qua được 10 bài test cơ bản.")
            print(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout if e.stdout else ""
        stderr = e.stderr if e.stderr else ""
        print("[FAIL] [BỊ LOẠI] Quá thời gian chạy thử (Timeout 60s). Nghi ngờ Agent bị lỗi vòng lặp vô hạn (Infinite Loop)!")
        print(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")
        return False
    except Exception as e:
        print(f"[FAIL] [BỊ LOẠI] Lỗi môi trường khi chạy test: {e}")
        return False

    # Nếu sống sót qua cả 2 bước, Agent mới thực sự hợp lệ
    print("[SUCCESS] [THÀNH CÔNG] Sanity Check vượt qua! Chuẩn bị cấp ngân sách chạy Giai đoạn 2.")
    return True

if __name__ == "__main__":
    # Test on the current LAMBDA.py as a baseline sanity check
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_lambda = os.path.join(repo_root, "LAMBDA.py")
    is_valid_agent(target_lambda)
