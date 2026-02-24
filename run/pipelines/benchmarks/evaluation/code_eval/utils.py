import subprocess
import re
import os
import tempfile

def extract_code(text: str) -> str:
    """Extracts code from a ```python ... ``` block."""
    match = re.search(r"```(?:python)?\n(.*?)\n```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()

def build_test_code(code_str: str, testcase_str: str, entry_point: str = None, bench_name: str = "humaneval") -> str:
    """Builds a complete code string that includes the solution and test cases."""
    test_code = ""
    if bench_name == "human-eval":
        test_code = f"{code_str}\n\n{testcase_str}\ncheck({entry_point})"
    else:
        raise NotImplementedError(f"Test code generation not implemented for benchmark: {bench_name}")
    
    return test_code
    
def run_single_test(code_str: str, timeout: float = 2.0) -> bool:
    """Run the generated code with the provided test cases and return whether all tests passed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        code_path = os.path.join(temp_dir, "solution.py")
        
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code_str)

        try:
            proc = subprocess.run(
                ["python", code_path],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            # If returncode is 0 and no assertion errors, tests passed
            if proc.returncode == 0:
                return True
            else:
                # Check if it's an assertion error
                # if "AssertionError" in proc.stderr or "assert" in proc.stderr.lower():
                #     return False
                # Other runtime errors also count as failure
                return False
        except (subprocess.TimeoutExpired, Exception):
            return False