from pysfrl.sim.simulator import Simulator
import json


class SimResult(object):
    @staticmethod
    def sim_result_to_json(sim: Simulator, file_path):    
        result_data = {}        
        for i in range(0, sim.time_step+1):            
            result_data[i] = {
                "step_width": sim.step_width,
                "states": sim.peds.states[i].tolist()
            }
        
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4)        
        return

    @staticmethod
    def summary_to_json(sim: Simulator, file_path, success):
        data = {}
        data["success"] = success
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return

    @staticmethod
    def result_info_to_json(sim: Simulator, file_path):
        pass
    