from pysfrl.sim.new_simulator import NewSimulator
import json


class SimResult(object):
    @staticmethod
    def sim_result_to_json(sim: NewSimulator, file_path):    
        result_data = {}        
        time = 0
        result_data[0] = {
            "step_width": time,
            "states": sim.peds.states[0].tolist()
        }
        
        for i in range(0, len(sim.step_width_list)):
            time += sim.step_width_list[i]
            result_data[i+1] = {
                "step_width": time,
                "states": sim.peds.states[i+1].tolist()
            }
        
        with open(file_path, 'w') as f:
            json.dump(result_data, f, indent=4)        
        return

    @staticmethod
    def summary_to_json(sim: NewSimulator, file_path, success):
        data = {}
        data["success"] = success
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        return
    