import cv2

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