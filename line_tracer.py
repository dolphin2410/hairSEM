from renderer import ImageRenderer, RenderTasks
from input_manager import InputManager, SubscriptionType
import geometrics

class LineTracer:
    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.start_point = None
        self.end_point = None
        self.lock = False

        input_manager.subscribe(SubscriptionType.LEFT_CLICK, self.on_click)  # subscribe click events

    def tick(self):
        if not self.lock and self.start_point is not None:
            self.end_point = self.input_manager.cursor_pos  # creates a temporary end_point at the cursor's position
        
        self.render_line()
    
    def on_click(self, x, y):
        """initializes the points when clicked"""

        if self.lock:
            return
        
        # when choosing the first point
        if self.start_point is None:  
            self.start_point = (x, y)
        
        # when choosing the second point
        else:  
            self.end_point = (x, y)
            self.lock = True

    def render_line(self):
        # This phrase catches two cases - one when two points are identical and one when both None
        if self.start_point == self.end_point:
            return
        
        extended_ends = self.extend_line()
        self.renderer.push_task(RenderTasks.DRAW_LINE, extended_ends)

    def extend_line(self):
        """Returns intercepts of a linear graph determined by two points"""

        linear_graph = geometrics.LinearGraph(self.start_point, self.end_point)

        return linear_graph.boundary_intercepts()
    
class LineTracerManager:
    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.old_tracers = []
        self.current_tracer = LineTracer(renderer, input_manager)

    def tick(self):
        for tracer in self.old_tracers:
            tracer.tick()

        self.current_tracer.tick()

        if self.current_tracer.lock:
            self.old_tracers.append(self.current_tracer)
            self.current_tracer = LineTracer(self.renderer, self.input_manager)

    def revert_last(self):
        if len(self.old_tracers) > 0:
            self.old_tracers.pop()