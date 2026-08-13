import logging

import msgpack
import socket
import numpy as np
from rl_tools.game_engine.ObservationInterface import ObservationInterface

MAX_RETRIES = 5

# Flat binary observation protocol. Layout must stay in sync with
# modules/ObservationCollector/ObservationCollector.gd and
# torch_files/Factory/NetworkFactory.py (AGENTS.md). Native little-endian.
MAGIC = 0x53
VERSION = 0x01
N_CELLS = 192  # 16x12 grid
N_CELL_FEATURES = 7
N_GLOBAL_FEATURES = 15
N_BUILDERS = 5  # GameData.MAX_BUILDERS
N_BUILDER_FEATURES = 6
N_BUILDINGS = 10
HEADER = 2  # magic + version


class UDPObservation(ObservationInterface):
    def __init__(self, logger: logging.Logger, ip: str, port: int):
        super().__init__(logger=logger)
        self._udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_client.bind((ip, port))
        self._udp_client.settimeout(1.0)
        self._logger.info(f"UDPObservation initialized and listening on {ip}:{port}")

    def _get_observation(self) -> bytes | None:
        """Retrieve the current observation from the UDP client."""
        for _ in range(MAX_RETRIES):
            try:
                data, _ = self._udp_client.recvfrom(65536)
                break
            except socket.timeout:
                self._logger.debug("No observation received within timeout period.")
        else:
            self._logger.error("Failed to receive observation after multiple attempts.")
            return None
        return data

    def close(self) -> None:
        """Close the underlying UDP socket, releasing the bound port."""
        try:
            self._udp_client.close()
        except OSError:
            pass

    def parse_observation(self, raw_observation: bytes | None) -> dict | None:
        """Parse the raw observation data if necessary."""
        if raw_observation is None:
            return None
        if (
            len(raw_observation) >= HEADER
            and raw_observation[0] == MAGIC
            and raw_observation[1] == VERSION
        ):
            return self._parse_flat(raw_observation)
        try:
            return msgpack.unpackb(raw_observation, raw=False)
        except msgpack.UnpackException:
            self._logger.error("Could not unpack observation data. Invalid format.")
            return None

    def _parse_flat(self, data: bytes) -> dict | None:
        # Writable buffer so torch.from_numpy in split_observation gets writable
        # arrays (avoids the non-writable-tensor warning).
        p = bytearray(data[HEADER:])
        off = 0

        def take_f32(n: int) -> np.ndarray:
            nonlocal off
            arr = np.frombuffer(p[off : off + n * 4], np.float32)
            off += n * 4
            return arr

        def take_u8(n: int) -> np.ndarray:
            nonlocal off
            arr = np.frombuffer(p[off : off + n], np.uint8)
            off += n
            return arr

        try:
            fields = take_f32(N_CELLS * N_CELL_FEATURES).reshape(
                N_CELLS, N_CELL_FEATURES
            )
            global_features = take_f32(N_GLOBAL_FEATURES)
            builders = take_f32(N_BUILDERS * N_BUILDER_FEATURES).reshape(
                N_BUILDERS, N_BUILDER_FEATURES
            )
            buildable_cells = take_u8(N_BUILDINGS * N_CELLS).reshape(
                N_BUILDINGS, N_CELLS
            )
            available_buildings = take_u8(N_BUILDINGS)
            moveable_cells = take_u8(N_BUILDERS * N_CELLS).reshape(N_BUILDERS, N_CELLS)
            available_builders = take_u8(N_BUILDERS)
            real_builders = take_u8(N_BUILDERS)
            available_skip = take_u8(1)[0]
            reward = float(take_f32(1)[0])
            done = bool(take_u8(1)[0])
            info_len = int(np.frombuffer(p[off : off + 4], np.uint32)[0])
            info = msgpack.unpackb(p[off + 4 : off + 4 + info_len], raw=False)
        except (ValueError, IndexError) as e:
            self._logger.error(f"Could not parse flat observation: {e}")
            return None

        return {
            "observation": {
                "fields": fields,
                "global": global_features,
                "builders": builders,
            },
            "action_mask": {
                "buildable_cells": buildable_cells,
                "available_buildings": available_buildings,
                "moveable_cells": moveable_cells,
                "available_builders": available_builders,
                "real_builders": real_builders,
                "available_skip": available_skip,
            },
            "reward": reward,
            "done": done,
            "info": info,
        }
