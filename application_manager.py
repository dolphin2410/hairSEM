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

from enum import Enum
from crop_manager import CropManager
from line_tracer import LineTracerManager
from renderer import ImageRenderer, RenderTasks
from input_manager import HairSEMEvents, InputManager

class ApplicationMode(Enum):
    LINE_TRACING = 0
    CROPPING = 1

class ApplicationManager:
    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.mode = ApplicationMode.LINE_TRACING

        self.line_tracer_manager = LineTracerManager(renderer, input_manager)
        self.crop_manager = CropManager(renderer, input_manager)

        self.line_tracer_manager.initialize()

    def handle_inputs(self):
        input_ev = self.input_manager.current_event

        if input_ev == HairSEMEvents.SWITCH_MODE:
            self.cleanup()
            self.mode = ApplicationMode((self.mode.value + 1) % len(ApplicationMode))
            self.initialize()

    def initialize(self):
        if self.mode == ApplicationMode.LINE_TRACING:
            self.line_tracer_manager.initialize()
        elif self.mode == ApplicationMode.CROPPING:
            self.crop_manager.initialize()

    def cleanup(self):
        if self.mode == ApplicationMode.LINE_TRACING:
            self.line_tracer_manager.cleanup()
        elif self.mode == ApplicationMode.CROPPING:
            self.crop_manager.cleanup() 

    def update(self):
        self.handle_inputs()

        if self.mode == ApplicationMode.LINE_TRACING:
            mode_text = "line_tracing"
            self.line_tracer_manager.update()
        elif self.mode == ApplicationMode.CROPPING:
            mode_text = "cropping"
            self.crop_manager.update()    

        self.renderer.push_task(RenderTasks.WRITE_TEXT, [f"mode : {mode_text}", (10, 30)])