from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.support import WORST_REL, load_fixture


@pytest.fixture
def grid12() -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
    return load_fixture("grid12")


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: Any
) -> None:
    if not WORST_REL:
        return
    terminalreporter.write_sep("=", "Golden worst relative error")
    for name, value in sorted(WORST_REL.items()):
        terminalreporter.write_line(f"{name}: {value:.6g}")
