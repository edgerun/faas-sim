from typing import Tuple, Dict, Optional

from srds import ParameterizedDistribution as PDist, BoundRejectionSampler, BufferedSampler


from sim.oracle.oracle import FetOracle, ResourceOracle

def extract_model_type(device: str):
    if not type(device) is str:
        return ''
    try:
        return device[:device.rindex('_')]
    except ValueError:
        return device

# TODO move this implementation of FetOracle into the interface definition with the name 'BufferedFetOracle'
class Raith21FetOracle(FetOracle):

    def __init__(self, execution_times: Dict[Tuple[str, str], Tuple[float, float, PDist]]):
        super().__init__()
        self.execution_times = execution_times
        self.execution_time_samplers = {
            k: BoundRejectionSampler(BufferedSampler(dist), xmin, xmax) for k, (xmin, xmax, dist) in
            execution_times.items()
        }

    def sample(self, host: str, image: str) -> Optional[float]:
        host_type = extract_model_type(host) if '_' in host else host
        image_key = image.split(':')[0]  # strip version number

        k = (host_type, image_key)
        if k not in self.execution_time_samplers:
            return None

        return self.execution_time_samplers[k].sample()


class Raith21ResourceOracle(ResourceOracle):

    def __init__(self, resources: Dict[Tuple[str, str], 'FunctionResourceCharacterization']):
        super().__init__()
        self.resources = resources

    def get_resources(self, host: str, image: str):
        host = extract_model_type(host) if '_' in host else host
        return self.resources.get((host, image), None)
