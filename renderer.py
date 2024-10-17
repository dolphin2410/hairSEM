from enum import Enum
import cv2
import numpy as np

class RenderTasks(Enum):
    DRAW_LINE = 0

class ImageRenderer:
    def __init__(self, raw_image: np.ndarray):
        self.raw_image = np.copy(raw_image)
        self.image = np.copy(raw_image)
        self.tasks = []
    
    def push_task(self, task_type, payload):
        self.tasks.append((task_type, payload))
    
    def handle_task(self, task_type, payload):
        if task_type == RenderTasks.DRAW_LINE:
            self.image = cv2.line(self.image, payload[0], payload[1], 255)

    def tick(self):
        while len(self.tasks) > 0:
            task_type, payload = self.tasks.pop(0)
            self.handle_task(task_type, payload)
        
    def reset(self):
        self.image = np.copy(self.raw_image)