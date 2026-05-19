import ast
import tempfile
import os
import importlib.util
import time
import multiprocessing
import traceback

def validate_ast(code: str) -> tuple[bool, str]:
    """Kiểm tra AST: Syntax hợp lệ và có class EvolutionStrategy"""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Lỗi cú pháp (Syntax Error) dòng {e.lineno}: {e.msg}"
        
    has_class = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'EvolutionStrategy':
            has_class = True
            
            # Kiểm tra method select_parent
            has_select = False
            for cls_node in node.body:
                if isinstance(cls_node, ast.FunctionDef) and cls_node.name == 'select_parent':
                    has_select = True
                    break
            
            if not has_select:
                return False, "Thiếu method `select_parent(self, archive: list) -> dict` trong class EvolutionStrategy"
                
    if not has_class:
        return False, "Thiếu class tên `EvolutionStrategy`"
        
    # Security: cấm import OS/SYS/Subprocess để tránh Agent thao tác nguy hiểm
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ['os', 'sys', 'subprocess', 'shutil', 'pathlib']:
                    return False, f"Bảo mật: Không được phép import module nguy hiểm: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module in ['os', 'sys', 'subprocess', 'shutil', 'pathlib']:
                return False, f"Bảo mật: Không được phép import module nguy hiểm: {node.module}"
                
    return True, "AST Hợp lệ"

def _run_sandbox(code: str, return_dict):
    """Chạy giả lập trong multiprocessing riêng biệt để tránh treo hệ thống (timeout)"""
    try:
        # Ghi file tạm
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w', encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name
            
        # Load module
        spec = importlib.util.spec_from_file_location("temp_strategy", temp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Instantiate
        strategy = module.EvolutionStrategy()
        
        # Mock archive
        mock_archive = [
            {'cycle': 1, 'score': 0.8, 'epiplexity_score': 1.0, 'children_count': 1},
            {'cycle': 2, 'score': 0.9, 'epiplexity_score': 1.1, 'children_count': 0},
            {'cycle': 3, 'score': 0.5, 'epiplexity_score': 1.5, 'children_count': 2},
        ]
        
        # Test method
        result = strategy.select_parent(mock_archive)
        
        if not isinstance(result, dict):
            return_dict['error'] = f"Kiểu trả về sai: kỳ vọng dict, nhận được {type(result)}"
        elif not result and mock_archive:
            return_dict['error'] = "Trả về dict rỗng dù archive có dữ liệu"
        else:
            return_dict['success'] = True
            
    except Exception as e:
        return_dict['error'] = f"Lỗi runtime: {str(e)}\n{traceback.format_exc()}"
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

def validate_strategy(code: str) -> tuple[bool, str]:
    """
    Validate toàn diện code chiến lược mới sinh ra.
    Trả về (is_valid, reason)
    """
    # 1. AST Check
    is_ast_ok, ast_msg = validate_ast(code)
    if not is_ast_ok:
        return False, ast_msg
        
    # 2. Runtime Check với Timeout (bảo vệ khỏi infinite loops)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    return_dict['success'] = False
    
    p = multiprocessing.Process(target=_run_sandbox, args=(code, return_dict))
    p.start()
    p.join(timeout=3.0) # 3 giây timeout
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Runtime Timeout: Quá trình chọn lọc mất quá 3s (Nghi ngờ Infinite Loop)"
        
    if return_dict.get('error'):
        return False, return_dict['error']
        
    if not return_dict.get('success', False):
        return False, "Lỗi không xác định khi chạy giả lập"
        
    return True, "Code chiến lược hoàn toàn hợp lệ!"
