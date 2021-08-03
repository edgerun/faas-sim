from sim.oracle.oracle import ResourceOracle


class LRTResourceOracle(ResourceOracle):
    def get_resources(self, host: str, image: str) -> 'FunctionResourceCharacterization':
        pass
