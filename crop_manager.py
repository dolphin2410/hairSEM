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

import random
import numpy as np
from renderer import ImageRenderer, RenderTasks
from input_manager import HairSEMEvents, InputManager, SubscriptionType
from settings import CROP_IMAGE_SIZE, X_SIZE, Y_SIZE
import uuid
import cv2

class CropManager:
    """128x128 crop을 위한 클래스"""

    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.start_point = None
        self.lock = False

    def initialize(self):
        self.input_manager.subscribe(SubscriptionType.LEFT_CLICK, self.on_click)

    def handle_inputs(self):
        """호출된 이벤트 처리"""

        input_ev = self.input_manager.current_event

        # CROP 진행

        if input_ev == HairSEMEvents.SAVE_CROPPED_IMAGE:
            self.save([self.renderer.raw_image.copy(), self.renderer.raw_mask_image.copy()])
            self.input_manager.current_event == HairSEMEvents.EXIT

    def cleanup(self):
        """초기 상태로 초기화"""

        self.start_point = None

    def update(self):
        """업데이트(tick)"""

        # 호출된 이벤트 처리
        self.handle_inputs()

        if self.start_point is None:
            return
        
        self.render_box()
    
    def on_click(self, x, y):
        """좌클릭 콜백"""

        if self.lock:
            return
        
        # when choosing the first point
        if self.start_point is None:  
            self.start_point = (x, y)
            self.lock = True

    def render_box(self):
        """crop 영역을 파란 상자로 렌더링"""

        start_x, start_y = self.start_point

        p2 = start_x, start_y + CROP_IMAGE_SIZE
        p3 = start_x + CROP_IMAGE_SIZE, start_y + CROP_IMAGE_SIZE
        p4 = start_x + CROP_IMAGE_SIZE, start_y

        self.renderer.push_task(RenderTasks.DRAW_LINE, [self.start_point, p2])
        self.renderer.push_task(RenderTasks.DRAW_LINE, [p2, p3])
        self.renderer.push_task(RenderTasks.DRAW_LINE, [p3, p4])
        self.renderer.push_task(RenderTasks.DRAW_LINE, [p4, self.start_point])

    def save(self, images):
        """저장한다"""

        # 딱히 네모 상자를 정하지 않았다면, 랜덤하게 500회 수행
        if self.start_point is None:
            for i in range(500):

                # 랜덤한 위치 선정
                x_rand = int(random.random() * (X_SIZE - CROP_IMAGE_SIZE - 1))
                y_rand = int(random.random() * (Y_SIZE - CROP_IMAGE_SIZE - 1))
                self.start_point = x_rand, y_rand

                # self.start_point 설정 후 재귀
                self.save(images)
            
            self.start_point = None
            return

        x, y = self.start_point
        cropped = []
        
        for image in images:
            cropped_image = []
            for delta_y in range(CROP_IMAGE_SIZE):
                row = image[y + delta_y][x:x + CROP_IMAGE_SIZE]
                cropped_image.append(row)
            cropped.append(np.array(cropped_image))

        # UUID 이름으로 crop한 파일 저장
        uuid_string = uuid.uuid4()
        cv2.imwrite(f"sem_cropped_images/images/{uuid_string}.jpg", cropped[0])
        cv2.imwrite(f"sem_cropped_images/segmentation-masks/{uuid_string}.jpg", cropped[1])