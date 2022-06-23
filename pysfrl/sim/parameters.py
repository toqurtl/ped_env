from enum import Enum

class DataIndex(Enum):
    px = ("px", 0)
    py = ("py", 1)
    vx = ("vx", 2)
    vy = ("vy", 3)
    gx = ("gx", 4)
    gy = ("gy", 5)
    distancing = ("distancing", 6)
    tau = ("tau", 7)     
    visible = ("visible", 8)       
    start_time = ("start_time", 9)    
    id = ("id", 10)
    finished = ("finished", 11)
    phase = ("phase", 12)
    final_phase = ("final_phase", 13)

    def __init__(self, str_name, index):
        self.str_name = str_name
        self.index = index

    @classmethod
    def state_range(cls):
        return

    