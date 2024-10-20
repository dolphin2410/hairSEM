import cv2
from enum import Enum

class HairSEMEvents(Enum):
    PASS = 0
    EXIT = 1
    REMOVE_PREVIOUS_LINE = 2

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
        
        return HairSEMEvents.PASS