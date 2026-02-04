import socket
from game_engine.ObservationInterface.ObservationInterface import ObservationInterface

MAX_RETRIES = 5

class UDPObservation(ObservationInterface):
    def __init__(self, ip: str, port: int):
        super().__init__()
        self._udp_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_client.bind((ip, port))
        self._udp_client.settimeout(0.5) 

    def _get_observation(self):
        """Retrieve the current observation from the UDP client."""
        for _ in range(MAX_RETRIES):
            try:
                data, _ = self._udp_client.recvfrom(4096)
                break
            except socket.timeout:
                print("No observation received within timeout period.")
        else:
            print("Failed to receive observation after multiple attempts.")
            return None
        return data