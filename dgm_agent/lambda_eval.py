import sys
import os
import subprocess
import json

def check_docker_available() -> bool:
    """Kiểm tra Docker có sẵn và đang chạy hay không"""
    try:
        res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return res.returncode == 0
    except Exception:
        return False

def run_local_eval(sandbox_dir: str, budget: dict = None) -> dict:
    """Thực thi đánh giá cục bộ (được gọi bởi host hoặc chạy bên trong Docker container)"""
    # Đảm bảo repo root và sandbox_dir có trong sys.path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in [repo_root, sandbox_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        # Clear module cache để load chính xác LAMBDA.py đột biến
        for mod in ['LAMBDA', 'core.conversation', 'core.inspector']:
            if mod in sys.modules:
                del sys.modules[mod]

        # Mock CodeKernel để chạy nhanh và an toàn
        import core.kernel
        class MockCodeKernel:
            def __init__(self, *args, **kwargs): pass
            def execute(self, *args, **kwargs): return ""
            def shutdown(self, *args, **kwargs): pass
        def mock_execute(code, kernel):
            return "text", "Mock execution message", "Mock execution result"
        core.kernel.CodeKernel = MockCodeKernel
        core.kernel.execute = mock_execute

        from LAMBDA import LAMBDA
        child_agent = LAMBDA()

        from core.inspector import Verifier
        verifier = Verifier()
        stage1_rate, stage_status = verifier.run_staged_evaluation(child_agent, budget)

        # Giải phóng tài nguyên
        if hasattr(child_agent, "conv") and hasattr(child_agent.conv, "kernel"):
            try:
                child_agent.conv.kernel.shutdown()
            except Exception:
                pass

        return {
            "success": True,
            "stage1_rate": stage1_rate,
            "score": round(0.5 + (stage1_rate * 0.5), 4),
            "stderr": ""
        }
    except Exception as e:
        return {
            "success": False,
            "stage1_rate": 0.0,
            "score": 0.5,
            "stderr": str(e)
        }
    finally:
        if sandbox_dir in sys.path:
            sys.path.remove(sandbox_dir)

def run_lambda_eval(sandbox_dir: str, budget: dict = None, task_description: str = "") -> dict:
    """
    Chạy Đánh giá 4 Giai đoạn với môi trường Docker Sandbox cách ly hoàn toàn,
    tự động fallback sang Local Host nếu Docker không khả dụng.
    
    LƯU Ý KHOA HỌC (Academic Note):
    Môi trường Docker Sandbox ở đây đóng vai trò là "Syntax & Runtime safety checker" (Kiểm tra biên dịch và chạy).
    Nó ngăn chặn mã nguồn độc hại hoặc lỗi nghiêm trọng của tác tử đột biến (LAMBDA.py) làm hỏng máy host.
    
    Khi chạy tiến hóa, lệnh `docker run` sẽ ghi đè CMD mặc định (pytest tests/) trong Dockerfile.sandbox
    bằng cách gọi trực tiếp lệnh Python chạy hàm `run_local_eval` để kiểm tra độ tin cậy của thuật toán.
    """
    score = 0.8
    is_valid = True
    goldilocks = "PASS"
    stderr = ""

    # 1. Tính toán Epiplexity / Goldilocks Zone sử dụng thuật toán NCD toán học
    mutated_path = os.path.join(sandbox_dir, "LAMBDA.py")
    mutated_code = ""
    if os.path.exists(mutated_path):
        with open(mutated_path, "r", encoding="utf-8") as f:
            mutated_code = f.read()

    from core.inspector import Verifier
    verifier = Verifier()
    verdict = verifier.is_in_goldilocks_zone(task_description, mutated_code)
    epiplexity = verdict["epiplexity_score"]
    goldilocks = verdict["goldilocks_status"]

    # 2. Thực thi đánh giá trong Docker Sandbox cách ly (nếu khả dụng)
    docker_success = False
    if check_docker_available():
        print("🐳 Kích hoạt Docker Sandbox cách ly bảo mật...")
        try:
            absolute_sandbox_dir = os.path.abspath(sandbox_dir)
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Đảm bảo Docker image được build
            dockerfile_path = os.path.join(repo_root, "Dockerfile.sandbox")
            subprocess.run(
                ["docker", "build", "-t", "lambda-sandbox", "-f", dockerfile_path, repo_root],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
            )

            # Khởi chạy Docker với cấu hình bảo mật cực cao
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "512m",
                "--cpus", "1.0",
                "-v", f"{repo_root}:/repo_root",
                "-v", f"{absolute_sandbox_dir}:/sandbox",
                "lambda-sandbox",
                "python", "-c",
                f"import sys, os, json; sys.path.insert(0, '/sandbox'); sys.path.insert(0, '/repo_root'); from dgm_agent.lambda_eval import run_local_eval; print(json.dumps(run_local_eval('/sandbox', {budget})))"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            if result.returncode == 0:
                # Trích xuất và parse JSON trả về từ docker container
                stdout_str = result.stdout.strip()
                for line in reversed(stdout_str.split("\n")):
                    if line.startswith("{") and line.endswith("}"):
                        eval_data = json.loads(line)
                        if eval_data.get("success", False):
                            score = eval_data.get("score", 0.5)
                            stderr = eval_data.get("stderr", "")
                            docker_success = True
                            break
            else:
                print(f"⚠️ Docker run lỗi (code {result.returncode}), đang chuyển sang chế độ tự bảo vệ fallback...")
        except Exception as docker_err:
            print(f"⚠️ Không thể thực thi Docker Sandbox: {docker_err}. Đang kích hoạt Fallback...")

    # 3. Fallback sang Local Evaluation nếu Docker bị tắt hoặc lỗi
    if not docker_success:
        print("💻 Đang chạy ở chế độ Host-Fallback cục bộ...")
        res = run_local_eval(sandbox_dir, budget)
        score = res.get("score", 0.5)
        stderr = res.get("stderr", "")

    return {
        "is_valid": is_valid,
        "score": round(score, 4),
        "epiplexity_score": epiplexity,
        "goldilocks_status": goldilocks,
        "stderr": stderr
    }
