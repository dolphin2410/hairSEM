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
    """
    이미지 렌더링을 위한 관리자
    """

    def __init__(self, raw_image: np.ndarray, raw_mask_image: np.ndarray, input_manager: InputManager):
        self.raw_image = np.copy(raw_image)
        self.raw_mask_image = np.copy(raw_mask_image)
        self.image_type = ImageType.RAW_IMAGE

        self.image = np.copy(raw_image)
        self.tasks = []
        self.input_manager = input_manager

    def push_task(self, task_type, payload):
        """
        렌더링 태스크를 등록한다 (업데이트 시 한번에 처리)
        """

        self.tasks.append((task_type, payload))
    
    def handle_task(self, task_type, payload):
        """
        등록된 렌더링 태스크를 처리한다
        """

        # 선을 선분을 그리는 작업
        if task_type == RenderTasks.DRAW_LINE:
            cv2.line(self.image, payload[0], payload[1], (255, 0, 0))
        
        # 텍스트를 그리는 작업
        elif task_type == RenderTasks.WRITE_TEXT:
            cv2.putText(self.image, payload[0], payload[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    def handle_inputs(self):
        """
        호출된 이벤트를 처리한다
        """

        input_ev = self.input_manager.current_event

        # 렌더링할 이미지의 종류를 바꾼다
        if input_ev == HairSEMEvents.SWITCH_IMAGE:
            self.image_type = ImageType((self.image_type.value + 1) % len(ImageType))

    def update(self):
        """
        현재 화면을 업데이트 한다
        """

        self.handle_inputs()

        # 렌더링 종류에 따라서 원본을 복사한다
        if self.image_type == ImageType.RAW_IMAGE:
            self.image = np.copy(self.raw_image)
        else:
            self.image = np.copy(self.raw_mask_image)

        # 등록된 렌더링 작업을 모두 수행한다
        while len(self.tasks) > 0:
            task_type, payload = self.tasks.pop(0)
            self.handle_task(task_type, payload)
        
        # 렌더링 후 이미지를 띄운다
        cv2.imshow('img_window', self.image)
