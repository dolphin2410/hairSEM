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

import cv2
from enum import Enum

class HairSEMEvents(Enum):
    PASS = 0
    EXIT = 1
    REMOVE_PREVIOUS_LINE = 2
    SAVE_CROPPED_IMAGE = 3

class SubscriptionType(Enum):
    LEFT_CLICK = 0

class InputManager:
    def __init__(self):
        self.cursor_pos = None
        self.lclick_watchers = []

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            for watcher in self.lclick_watchers:
                watcher(x, y)

    def subscribe(self, subscription_type: SubscriptionType, f):
        if subscription_type == SubscriptionType.LEFT_CLICK:
            self.lclick_watchers.append(f)
            return len(self.lclick_watchers) - 1
    
    def unsubscribe(self, subscription_type: SubscriptionType, idx):
        if subscription_type == SubscriptionType.LEFT_CLICK:
            self.lclick_watchers.pop(idx)

    def get_keyevent(self):
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return HairSEMEvents.EXIT

        if key & 0xFF == 27:
            return HairSEMEvents.REMOVE_PREVIOUS_LINE
        
        if key & 0xFF == ord('s'):
            return HairSEMEvents.SAVE_CROPPED_IMAGE
        
        return HairSEMEvents.PASS