from __future__ import annotations
from game_engine.HeadlessGameEngine.HeadlessGameEngine import HeadlessGameEngine
from game_engine.HeadlessGameEngine.GodotHeadless.GodotHeadless import GodotHeadless


class HeadlessGameEngineFactory:
    instance: HeadlessGameEngineFactory | None = None

    mapping = {
        HeadlessGameEngine.GameEngineType.GODOT: GodotHeadless,
    }

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super(HeadlessGameEngineFactory, cls).__new__(cls)
        return cls.instance

    def create(
        self,
        game_engine_type: HeadlessGameEngine.GameEngineType | None = None,
        instance_id: int = 0,
        run_args: list | None = None,
        run_kwargs: dict | None = None,
        **kwargs,
    ):
        game_engine_type = (
            game_engine_type
            if game_engine_type
            else HeadlessGameEngine.GameEngineType.GODOT
        )
        run_args = run_args if run_args else []
        run_kwargs = run_kwargs if run_kwargs else {}
        kwargs = kwargs
        if game_engine_type in self.mapping:
            return self.mapping[game_engine_type](
                instance_id=instance_id,
                run_args=run_args,
                run_kwargs=run_kwargs,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported game engine type: {game_engine_type}")
