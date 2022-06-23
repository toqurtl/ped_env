from pysfrl.config.config import SimulationConfig

class NewSimulator(object):
    def __init__(self, config: SimulationConfig):
        # configuration
        self.scene_config = config.scene_config
        self.force_config = config.force_config

        # initialization(config 정보 바탕으로 초기정보 생성)
        self._initialize()
        # Simulation 


    def _initialize(self):
        pass

    def step(self):
        pass
