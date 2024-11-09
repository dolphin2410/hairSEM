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

import cv2
from application_manager import ApplicationManager
from input_manager import InputManager, HairSEMEvents
from renderer import ImageRenderer
from settings import X_SIZE, Y_SIZE

# 이미지 불러오기

raw_image_data = cv2.imread("./sem_images/images/SEM_C_PRESSURE.jpg")  # SEM 이미지 불러오기
raw_mask_data = cv2.imread("./sem_images/segmentation-masks/SEM_C_PRESSURE.jpg")  # SEM 마스크 이미지 불러오기

resized_image_data = cv2.resize(raw_image_data, (X_SIZE, Y_SIZE))
resized_mask_data = cv2.resize(raw_mask_data, (X_SIZE, Y_SIZE))

input_manager = InputManager()  # 키보드, 마우스 입력을 위한 클래스 초기화
renderer = ImageRenderer(resized_image_data, raw_mask_data, input_manager)  # 이미지 렌더링을 위한 클래스 초기화

# 애플리케이션 객체 초기화

application_manager = ApplicationManager(renderer, input_manager)

# 마우스 입력 콜백 등록을 위한 초기 이미지 렌더링
cv2.imshow('img_window', resized_image_data)
cv2.setMouseCallback('img_window', input_manager.on_mouse)

while True:
    # 앱 업데이트 (tick)
    application_manager.update()
    
    # 탈출 이벤트가 호출되었다면 프로그램 종료
    if input_manager.current_event == HairSEMEvents.EXIT:
        break

cv2.destroyAllWindows()