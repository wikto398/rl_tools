import torch


class RunningMeanStd:
    """Welford-online running mean/variance accumulator over a leading batch dim.

    Stats are stored per-feature over the trailing dimension of the flattened
    input (shape ``(N, F)``). ``var`` holds the population variance and
    defaults to ``ones`` so that normalization is an identity before any
    update.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = 0

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.detach().float()
        if batch.shape[0] == 0:
            return
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total_count)
        m2_a = self.var * self.count
        m2_b = batch_var * batch_count
        m2 = m2_a + m2_b + delta.pow(2) * (self.count * batch_count / total_count)
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"].to(torch.float32).cpu()
        self.var = state["var"].to(torch.float32).cpu()
        self.count = int(state["count"])
