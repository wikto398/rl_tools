from __future__ import annotations


class BudgetSchedule:
    """Per-generation step budget for evolutionary children.

    Children of early generations get a small budget (cheap probes to prune
    bad genomes); later generations get larger budgets for the survivors.
    """

    def __init__(self, start: int, end: int, curve: str = "geometric"):
        self.start = int(start)
        self.end = int(end)
        self.curve = curve
        if self.start <= 0 or self.end < self.start:
            raise ValueError(
                f"budget start/end must satisfy 0 < start <= end, got {start}/{end}"
            )
        if curve not in ("geometric", "linear"):
            raise ValueError(f"unknown curve: {curve}")

    @classmethod
    def geometric(cls, start: int, end: int) -> BudgetSchedule:
        return cls(start, end, curve="geometric")

    @classmethod
    def linear(cls, start: int, end: int) -> BudgetSchedule:
        return cls(start, end, curve="linear")

    def __call__(self, generation: int, total: int) -> int:
        if total <= 1:
            return self.end
        if generation >= total:
            generation = total - 1
        if self.curve == "geometric":
            ratio = (self.end / self.start) ** (1.0 / (total - 1))
            return round(self.start * (ratio**generation))
        frac = generation / (total - 1)
        return round(self.start + (self.end - self.start) * frac)
