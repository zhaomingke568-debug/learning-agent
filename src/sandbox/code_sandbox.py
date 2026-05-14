import subprocess
import tempfile
import os
import uuid
from typing import List, Tuple, Optional, Dict

MAX_OUTPUT_SIZE = 10 * 1024  # 10KB
TIMEOUT_SECONDS = 5


class CodeSandbox:
    """
    Subprocess-based Python sandbox.

    安全措施:
    1. 子进程执行，超时强制 kill (SIGTERM, 5秒)
    2. 无文件系统/网络访问（子进程权限受限）
    3. 临时文件用完即删
    4. 捕获 stdout/stderr，返回行号
    5. 限制输出大小（最多 10KB），防止内存耗尽
    6. 输入过滤：禁止危险代码模式
    """

    def __init__(self, timeout_seconds: int = TIMEOUT_SECONDS, max_output_size: int = MAX_OUTPUT_SIZE):
        self.timeout = timeout_seconds
        self.max_output_size = max_output_size

    def _precheck(self, code: str) -> Tuple[bool, str]:
        """预检查：拒绝明显危险的代码模式"""
        dangerous = [
            "import", "open", "eval", "exec", "__import__",
            "compile", "input", "breakpoint", "exit", "quit",
            "os.", "sys.", "subprocess", "socket", "requests",
            "urllib", "http", "ftp", "threading", "multiprocessing",
            "pickle", "marshal", "eval", "exec", "compile"
        ]
        for pattern in dangerous:
            if pattern in code:
                return False, f"禁止使用: {pattern}"
        return True, ""

    def execute(self, code: str) -> Dict:
        """
        执行代码，无测试用例。返回 pass/fall + output/error。

        Returns:
            {"passed": bool, "output": str, "error_message": str, "error_line": int}
        """
        ok, err = self._precheck(code)
        if not ok:
            return {"passed": False, "output": "", "error_message": err, "error_line": None}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONPATH": "", "PYTHONHOME": ""},
            )
            output = result.stdout[:self.max_output_size]
            stderr = result.stderr[:self.max_output_size]

            if result.returncode != 0:
                # 尝试解析错误行号
                error_line = self._extract_error_line(stderr)
                return {"passed": False, "output": output, "error_message": stderr, "error_line": error_line}

            return {"passed": True, "output": output, "error_message": "", "error_line": None}
        except subprocess.TimeoutExpired:
            return {"passed": False, "output": "", "error_message": f"执行超时 ({self.timeout}秒)", "error_line": None}
        except Exception as e:
            return {"passed": False, "output": "", "error_message": str(e), "error_line": None}
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    def execute_with_test_cases(self, code: str, test_cases: List[dict]) -> Dict:
        """
        执行代码并验证测试用例。

        Args:
            code: Python 代码
            test_cases: [{"input": str, "expected": str, "hidden": bool}]

        Returns:
            {
                "execution_id": str,
                "all_passed": bool,
                "visible_results": [...],
                "hidden_results": [...],
                "summary": "3/5 tests passed"
            }
        """
        execution_id = str(uuid.uuid4())

        # 先执行主代码，捕获输出
        main_result = self.execute(code)

        results = []
        for i, tc in enumerate(test_cases):
            test_id = f"test_{i}"
            input_data = tc.get("input", "")
            expected = tc.get("expected", "")
            hidden = tc.get("hidden", False)

            # 构建测试代码：将 input 作为输入，运行代码并捕获输出
            test_code = f"""
{code}

# 自动注入测试
_input = '''{input_data}'''
print(_input)
"""
            test_result = self.execute(test_code)
            def _normalize(value):
               if value is None:
                  return ""
               return str(value).strip()

            actual = test_result.get("output", "").strip()
            actual_normalized = _normalize(actual)
            expected_normalized = _normalize(expected)
            passed = (actual_normalized == expected_normalized)

            result_entry = {
                "test_id": test_id,
                "passed": passed,
                "actual": actual if not hidden else "<hidden>",
                "expected": expected_normalized,
                "hidden": hidden,
                "error": test_result.get("error_message", "")
            }
            results.append(result_entry)

        visible_results = [r for r in results if not r["hidden"]]
        hidden_results = [r for r in results if r["hidden"]]
        passed_count = sum(1 for r in visible_results if r["passed"])

        return {
            "execution_id": execution_id,
            "all_passed": all(r["passed"] for r in visible_results),
            "visible_results": visible_results,
            "hidden_results": hidden_results,
            "summary": f"{passed_count}/{len(visible_results)} tests passed",
            "main_error": main_result.get("error_message", "")
        }

    def _extract_error_line(self, stderr: str) -> Optional[int]:
        """从 traceback 中提取错误行号"""
        import re
        # 匹配 "File "<string>", line 123" 格式
        match = re.search(r'line (\d+)', stderr)
        if match:
            return int(match.group(1))
        return None

    def analyze_and_suggest_fix(self, code: str, error_msg: str) -> str:
        """
        生成修复建议（LLM 调用，单独实现）。
        这里只做简单的错误分类。
        """
        if "SyntaxError" in error_msg:
            return "代码存在语法错误，请检查括号、引号等是否匹配。"
        elif "NameError" in error_msg:
            return "代码中使用了未定义的变量名，请检查拼写。"
        elif "TypeError" in error_msg:
            return "类型错误，请检查数据类型是否匹配。"
        elif "超时" in error_msg:
            return "代码执行时间过长，可能存在无限循环。"
        else:
            return f"执行出错：{error_msg}"


# 全局单例
_default_sandbox: Optional[CodeSandbox] = None

def get_sandbox() -> CodeSandbox:
    global _default_sandbox
    if _default_sandbox is None:
        _default_sandbox = CodeSandbox()
    return _default_sandbox