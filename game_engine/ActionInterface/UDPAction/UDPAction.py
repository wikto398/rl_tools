import logging
import socket

from rl_tools.game_engine.ActionInterface import ActionInterface


class UDPAction(ActionInterface):
    def __init__(self, logger: logging.Logger | None, ip: str, port: int):
        super().__init__(logger=logger)
        self.ip = ip
        self.port = port
        self._udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_client.connect((self.ip, self.port))
        self._udp_client.settimeout(1.0)
        self._logger.info(
            f"UDPAction initialized and connected to target {self.ip}:{self.port}"
        )

    def _send_action(self, action):
        action_bytes = self._serialize_action(action)
        self._udp_client.send(action_bytes)

    def _serialize_action(self, action):
        return bytearray(action)
