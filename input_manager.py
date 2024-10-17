import cv2

from events import HairSEMEvents

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
    
    def subscribe_lclick(self, f):
        self.lclick_watchers.append(f)
        return len(self.lclick_watchers) - 1
    
    def unsubscribe_lclick(self, idx):
        self.lclick_watchers.pop(idx)

    def get_keyevent(self):
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            return HairSEMEvents.EXIT

        if key & 0xFF == 27:
            return HairSEMEvents.REMOVE_PREVIOUS_LINE
        
        return HairSEMEvents.PASS