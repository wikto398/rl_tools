import socket
from game_engine.ActionInterface.ActionInterface import ActionInterface

class UDPAction(ActionInterface):
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self._udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_client.connect((self.ip, self.port))
        self._udp_client.settimeout(0.5) 

    def _send_action(self, action):
        self._udp_client.send(action)
