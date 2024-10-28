#    hairSEM aims to quantify hair damage based on a SEM image
#    Copyright (C) 2024 dolphin2410
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

from enum import Enum
import cv2
from matplotlib import pyplot as plt
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

    def update(self):
        """Reset and Update"""

        self.image = np.copy(self.raw_image)
        while len(self.tasks) > 0:
            task_type, payload = self.tasks.pop(0)
            self.handle_task(task_type, payload)