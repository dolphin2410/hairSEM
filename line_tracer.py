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

from renderer import ImageRenderer, RenderTasks
from input_manager import HairSEMEvents, InputManager, SubscriptionType
import geometrics

class LineTracer:
    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.start_point = None
        self.end_point = None
        self.lock = False

    def initialize(self):
        self.subscription_id = self.input_manager.subscribe(SubscriptionType.LEFT_CLICK, self.on_click)

    def update(self):
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
            self.input_manager.unsubscribe(SubscriptionType.LEFT_CLICK, self.subscription_id)

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

    def initialize(self):
        self.current_tracer.initialize()

    def handle_inputs(self):
        input_ev = self.input_manager.current_event
        if input_ev == HairSEMEvents.REMOVE_PREVIOUS_LINE:
            self.revert_last()

    def cleanup(self):
        self.old_tracers = []
        self.current_tracer = LineTracer(self.renderer, self.input_manager)

    def update(self):
        self.handle_inputs()
            
        for tracer in self.old_tracers:
            tracer.update()

        self.current_tracer.update()

        if self.current_tracer.lock:
            self.old_tracers.append(self.current_tracer)
            self.current_tracer = LineTracer(self.renderer, self.input_manager)
            self.current_tracer.initialize()

    def revert_last(self):
        if len(self.old_tracers) > 0:
            self.old_tracers.pop()