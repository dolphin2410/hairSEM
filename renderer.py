#    hairSEM aims to quantify hair damage using a SEM image
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

from input_manager import HairSEMEvents, InputManager

class RenderTasks(Enum):
    DRAW_LINE = 0
    WRITE_TEXT = 1

class ImageType(Enum):
    RAW_IMAGE = 0
    RAW_MASK_IMAGE = 1

class ImageRenderer:
    def __init__(self, raw_image: np.ndarray, raw_mask_image: np.ndarray, input_manager: InputManager):
        self.raw_image = np.copy(raw_image)
        self.raw_mask_image = np.copy(raw_mask_image)
        self.image_type = ImageType.RAW_IMAGE

        self.image = np.copy(raw_image)
        self.tasks = []
        self.input_manager = input_manager

    def switch(self):
        self.image_type = ImageType((self.image_type.value + 1) % len(ImageType))
    
    def push_task(self, task_type, payload):
        self.tasks.append((task_type, payload))
    
    def handle_task(self, task_type, payload):
        if task_type == RenderTasks.DRAW_LINE:
            cv2.line(self.image, payload[0], payload[1], (255, 0, 0))
        elif task_type == RenderTasks.WRITE_TEXT:
            cv2.putText(self.image, payload[0], payload[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def handle_inputs(self):
        input_ev = self.input_manager.current_event

        if input_ev == HairSEMEvents.SWITCH_IMAGE:
            self.switch()

    def update(self):
        """Reset and Update"""
        self.handle_inputs()

        if self.image_type == ImageType.RAW_IMAGE:
            self.image = np.copy(self.raw_image)
        else:
            self.image = np.copy(self.raw_mask_image)

        while len(self.tasks) > 0:
            task_type, payload = self.tasks.pop(0)
            self.handle_task(task_type, payload)
        
        cv2.imshow('img_window', self.image)
