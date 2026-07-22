from rl.Callback import Callback


class NetworkSaveCallback(Callback):
    def __init__(self, save_path: str):
        super().__init__()
        self.save_path = save_path

    def on_train_end(self) -> None:
        if self.agent is None:
            raise ValueError(
                "Agent is not set. Ensure the callback is attached to an agent."
            )
        self.agent.info(f"NetworkSaveCallback: Saving network to {self.save_path}")

        self.agent.save(self.save_path)
