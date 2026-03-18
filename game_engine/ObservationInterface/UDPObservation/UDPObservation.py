import logging

import msgpack
import socket
import json
from game_engine.ObservationInterface.ObservationInterface import ObservationInterface

MAX_RETRIES = 5

class UDPObservation(ObservationInterface):
    def __init__(self, logger: logging.Logger, ip: str, port: int):
        super().__init__(logger=logger)
        self._udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_client.bind((ip, port))
        self._udp_client.settimeout(0.5)
        self._logger.info(f"UDPObservation initialized and listening on {ip}:{port}")

    def _get_observation(self) -> bytes | None:
        """Retrieve the current observation from the UDP client."""
        for _ in range(MAX_RETRIES):
            try:
                data, _ = self._udp_client.recvfrom(4096)
                break
            except socket.timeout:
                self._logger.debug("No observation received within timeout period.")
        else:
            self._logger.error("Failed to receive observation after multiple attempts.")
            return None
        return data

    def parse_observation(self, raw_observation: bytes) -> dict | bytes | list:
        """Parse the raw observation data if necessary."""
        try:
            return msgpack.unpackb(raw_observation, raw=False)
        except msgpack.UnpackException:
            self._logger.error("Received observation is not valid MessagePack. Returning raw data.")
            return raw_observation
