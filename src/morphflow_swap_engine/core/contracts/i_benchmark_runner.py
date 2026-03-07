from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..value_objects.benchmark_case import BenchmarkCase
from ..value_objects.benchmark_run import BenchmarkRun


class IBenchmarkRunner(ABC):
    """Executes benchmark cases and returns scored results."""

    @abstractmethod
    def run(self, cases: List[BenchmarkCase]) -> List[BenchmarkRun]:
        """Run all cases and return one BenchmarkRun per case."""
        ...
