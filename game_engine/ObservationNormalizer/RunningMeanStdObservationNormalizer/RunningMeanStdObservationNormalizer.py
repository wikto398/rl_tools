from typing import Any

from tensordict import TensorDict

from rl_tools.game_engine.ObservationNormalizer import ObservationNormalizer
from rl_tools.game_engine.RunningMeanStd import RunningMeanStd


class RunningMeanStdObservationNormalizer(ObservationNormalizer):
    """RunningMeanStd normalization over each observation entry.

    Maintains one ``RunningMeanStd`` per observation key (``fields``,
    ``global``, ``builders``) over the trailing feature dimension. ``normalize``
    never mutates the input and returns a fresh ``TensorDict`` (safe for
    buffered observations). Before any stats are available normalization is the
    identity.
    """

    def __init__(
        self,
        clip_obs: float | None = 5.0,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_obs = clip_obs
        self.epsilon = epsilon
        self.training = True
        self._rms: dict[str, RunningMeanStd] = {}

    def _rms_for(self, key: str, n_features: int) -> RunningMeanStd:
        rms = self._rms.get(key)
        if rms is None:
            rms = RunningMeanStd((n_features,), epsilon=self.epsilon)
            self._rms[key] = rms
        return rms

    def update(self, observation: TensorDict) -> None:
        if not self.training:
            return
        for key, value in observation.items():
            flat = value.reshape(-1, value.shape[-1]).detach().float()
            self._rms_for(key, flat.shape[-1]).update(flat)

    def _normalize(self, observation: TensorDict) -> TensorDict:
        normalized: dict = {}
        for key, value in observation.items():
            rms = self._rms.get(key)
            if rms is None or rms.count == 0:
                normalized[key] = value
                continue
            mean = rms.mean.to(value.device)
            var = rms.var.to(value.device)
            out = (value - mean) / (var + self.epsilon).sqrt()
            if self.clip_obs is not None:
                out = out.clamp(-self.clip_obs, self.clip_obs)
            normalized[key] = out
        return TensorDict(normalized, batch_size=observation.batch_size)

    def state_dict(self) -> dict[str, Any]:
        return {key: rms.state_dict() for key, rms in self._rms.items()}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        self._rms = {}
        if not state:
            return
        for key, rms_state in state.items():
            rms = RunningMeanStd(epsilon=self.epsilon)
            rms.load_state_dict(rms_state)
            self._rms[key] = rms
