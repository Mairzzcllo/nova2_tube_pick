"""运行 debug/ 目录下的离线冒烟测试脚本。

用法（必须在仓库根目录或任意目录均可，脚本会自动定位仓库根目录）：
    ~/miniconda3/envs/Dobot-Sim/bin/python debug/run_smoke_tests.py

约定：
    debug/test_<module>.py       离线测试，默认运行，内部用 assert 判断
    debug/test_<module>_hw.py    需要真实硬件的测试，默认跳过
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEBUG_DIR = ROOT / "debug"


def discover_tests(pattern: str, include_hardware: bool) -> list[Path]:
    """按文件名模式收集测试脚本。"""

    tests = sorted(DEBUG_DIR.glob(pattern))

    if not include_hardware:
        tests = [path for path in tests if not path.stem.endswith("_hw")]

    return tests


def run_test(path: Path) -> bool:
    """用当前 Python 解释器执行单个测试脚本。"""

    print(f"[RUN ] {path.relative_to(ROOT)}")

    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
    )

    if result.returncode == 0:
        print(f"[PASS] {path.relative_to(ROOT)}")
        return True

    print(f"[FAIL] {path.relative_to(ROOT)} (exit={result.returncode})")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 debug 目录下的冒烟测试")
    parser.add_argument(
        "--pattern",
        default="test_*.py",
        help="测试文件匹配模式，默认 test_*.py",
    )
    parser.add_argument(
        "--include-hw",
        action="store_true",
        help="同时运行需要硬件的 *_hw.py 测试",
    )
    args = parser.parse_args()

    tests = discover_tests(args.pattern, args.include_hw)

    if not tests:
        print("未找到测试脚本。")
        return 1

    passed = []
    failed = []

    for path in tests:
        if run_test(path):
            passed.append(path)
        else:
            failed.append(path)

    print("-" * 60)
    print(f"通过 {len(passed)} 个，失败 {len(failed)} 个")

    for path in failed:
        print(f"  FAIL: {path.relative_to(ROOT)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
