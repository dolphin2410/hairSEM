from cv_layers import ImageRenderer
from input_manager import InputManager
import geometrics
from settings import X_SIZE, Y_SIZE

class LineTracer:
    def __init__(self, renderer: ImageRenderer, input_manager: InputManager, start_point):
        self.renderer = renderer
        self.input_manager = input_manager
        self.start_point = start_point
        self.end_point = None
        self.lock = False

        input_manager.subscribe_lclick(lambda x, y: self.handle_click(x, y))

    def tick(self):
        if not self.lock and self.start_point is not None:
            self.end_point = self.input_manager.pos
        
        self.render_line()
    
    def handle_click(self, x, y):
        if self.start_point is None:
            self.start_point = (x, y)
        elif not self.lock:
            self.end_point = (x, y)
            self.lock = True

    def render_line(self):
        if self.start_point == self.end_point:
            return
        
        extended_ends = self.extend_line()
        self.renderer.push_task("draw_line", extended_ends)
    
    def extend_line(self):
        linear_graph = geometrics.LinearGraph(self.start_point, self.end_point)

        return linear_graph.boundary_intercepts()