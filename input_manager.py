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
from enum import Enum
import uuid

class HairSEMEvents(Enum):
    PASS = 0
    EXIT = 1
    REMOVE_PREVIOUS_LINE = 2
    SAVE_CROPPED_IMAGE = 3
    SWITCH_MODE = 4
    SWITCH_IMAGE = 5
    ANALYZE = 6

class SubscriptionType(Enum):
    LEFT_CLICK = 0

class InputManager:
    """입력 관리자"""

    def __init__(self):
        self.cursor_pos = None
        self.lclick_watchers = {}
        self.current_event = HairSEMEvents.PASS

    def on_mouse(self, event, x, y, flags, param):
        """cv2에게 전달할 마우스 입력 콜백"""

        # 커서의 위치
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pos = (x, y)

        # 마우스 우클릭 처리
        if event == cv2.EVENT_LBUTTONDOWN:

            # 우클릭 리스너 처리
            for watcher in self.lclick_watchers.values():
                watcher(x, y)

    def subscribe(self, subscription_type: SubscriptionType, f):
        """특정 이벤트의 리스너를 등록한다 그리고 등록된 UUID를 반환한다"""

        if subscription_type == SubscriptionType.LEFT_CLICK:
            uniqueId = uuid.uuid4()
            self.lclick_watchers[uniqueId] = f
            return uniqueId
    
    def unsubscribe(self, subscription_type: SubscriptionType, uniqueId):
        """특정 이벤트의 리스너를 등록 해제한다"""

        if subscription_type == SubscriptionType.LEFT_CLICK:
            del self.lclick_watchers[uniqueId]

    def update(self):
        """키보드 입력을 받아서 이벤트를 호출한다"""

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            self.current_event = HairSEMEvents.EXIT
        elif key & 0xFF == 27:
            self.current_event = HairSEMEvents.REMOVE_PREVIOUS_LINE
        elif key & 0xFF == ord('s'):
            self.current_event = HairSEMEvents.SAVE_CROPPED_IMAGE
        elif key & 0xFF == ord('m'):
            self.current_event = HairSEMEvents.SWITCH_MODE
        elif key & 0xFF == ord('i'):
            self.current_event = HairSEMEvents.SWITCH_IMAGE
        elif key & 0xFF == ord('a'):
            self.current_event = HairSEMEvents.ANALYZE
        else:
            self.current_event = HairSEMEvents.PASS