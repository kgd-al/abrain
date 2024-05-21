import importlib
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("file", [
    pytest.param(p, id=str(p)) for p in
    Path("examples").glob("**/*.py")
])
def test_run_examples(file, capsys):
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file.stem] = module
    spec.loader.exec_module(module)

    if hasattr(module, "main"):
        with capsys.disabled():
            module.main(is_test=True)
