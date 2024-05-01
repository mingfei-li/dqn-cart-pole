from collections import deque
from torch.utils.tensorboard import SummaryWriter
from statistics import mean

class Logger():
    def __init__(self, run_id):
        self.scalar_buffer = {}
        self.writer = SummaryWriter(log_dir=f"results/logs/{run_id}")

    def add_scalar(self, key, value):
        if key not in self.scalar_buffer.keys():
            self.scalar_buffer[key] = deque(maxlen=200)
        self.scalar_buffer[key].append(value)
    
    def flush(self, t):
        for key, value in self.scalar_buffer.items():
            self.writer.add_scalar(key, mean(value), t)
        self.writer.flush()