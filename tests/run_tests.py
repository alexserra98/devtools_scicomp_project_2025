import sys
import subprocess
import argparse
from pathlib import Path


def setup_python_path():
    """Add src directory to Python path."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests."""
    print("Running unit tests...")

    cmd = ["python", "-m", "pytest", "tests/unit/", "-v" if verbose else ""]
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    # Remove empty strings
    cmd = [c for c in cmd if c]

    return subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("Running integration tests...")

    cmd = ["python", "-m", "pytest", "tests/integration/", "-v" if verbose else ""]
    cmd = [c for c in cmd if c]

    return subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    print("Running all tests...")

    cmd = ["python", "-m", "pytest", "tests/", "-v" if verbose else ""]
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    cmd = [c for c in cmd if c]

    return subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test."""
    print(f"Running specific test: {test_path}")

    cmd = ["python", "-m", "pytest", test_path, "-v" if verbose else ""]
    cmd = [c for c in cmd if c]

    return subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def check_test_environment():
    """Check if the test environment is properly set up."""
    print("Checking test environment...")

    # Check if pytest is available
    try:
        import pytest

        print(f"✓ pytest available (version {pytest.__version__})")
    except ImportError:
        print("✗ pytest not available. Install with: pip install pytest")
        return False

    # Check if source code is available
    setup_python_path()
    try:
        import importlib.util

        modules = ["attention_implementations", "transformer_model", "benchmark"]
        for module_name in modules:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                print(f"✗ Module {module_name} not available")
                return False
        print("✓ Source modules available")
    except Exception as e:
        print(f"✗ Source modules not available: {e}")
        return False

    # Check if PyTorch is available
    try:
        import torch

        print(f"✓ PyTorch available (version {torch.__version__})")
    except ImportError:
        print("✗ PyTorch not available")
        return False

    print("✓ Test environment ready")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for the transformer project"
    )
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "all", "check", "specific"],
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--test-path", help="Specific test file or test to run (for test_type=specific)"
    )

    args = parser.parse_args()

    # Setup environment
    setup_python_path()

    if args.test_type == "check":
        success = check_test_environment()
        sys.exit(0 if success else 1)

    elif args.test_type == "unit":
        result = run_unit_tests(args.verbose, args.coverage)

    elif args.test_type == "integration":
        result = run_integration_tests(args.verbose)

    elif args.test_type == "all":
        result = run_all_tests(args.verbose, args.coverage)

    elif args.test_type == "specific":
        if not args.test_path:
            print("Error: --test-path required when test_type=specific")
            sys.exit(1)
        result = run_specific_test(args.test_path, args.verbose)

    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
