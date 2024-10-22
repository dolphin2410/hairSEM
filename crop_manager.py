import numpy as np
from renderer import ImageRenderer, RenderTasks
from input_manager import InputManager, SubscriptionType
from settings import CROP_IMAGE_SIZE
import uuid
import cv2

class CropManager:
    def __init__(self, renderer: ImageRenderer, input_manager: InputManager):
        self.renderer = renderer
        self.input_manager = input_manager
        self.start_point = None
        self.lock = False

        input_manager.subscribe(SubscriptionType.LEFT_CLICK, self.on_click)  # subscribe click events

    def update(self):
        if self.start_point is None:
            return
        
        self.render_box()
    
    def on_click(self, x, y):
        """initializes the points when clicked"""

        if self.lock:
            return
        
        # when choosing the first point
        if self.start_point is None:  
            self.start_point = (x, y)
            self.lock = True

    def render_box(self):
        start_x, start_y = self.start_point

        p2 = start_x, start_y + CROP_IMAGE_SIZE
        p3 = start_x + CROP_IMAGE_SIZE, start_y + CROP_IMAGE_SIZE
        p4 = start_x + CROP_IMAGE_SIZE, start_y

        self.renderer.push_task(RenderTasks.DRAW_LINE, [self.start_point, p2])
        self.renderer.push_task(RenderTasks.DRAW_LINE, [p2, p3])
        self.renderer.push_task(RenderTasks.DRAW_LINE, [p3, p4])
        self.renderer.push_task(RenderTasks.DRAW_LINE, [p4, self.start_point])

    def save(self, images):
        if not self.lock:
            return

        x, y = self.start_point
        cropped = []
        
        for image in images:
            cropped_image = []
            for delta_y in range(CROP_IMAGE_SIZE):
                row = image[y + delta_y][x:x + CROP_IMAGE_SIZE]
                cropped_image.append(row)
            cropped.append(np.array(cropped_image))
    
        uuid_string = uuid.uuid4()
        cv2.imwrite(f"sem_cropped_images/images/{uuid_string}.jpg", cropped[0])
        cv2.imwrite(f"sem_cropped_images/segmentation-masks/{uuid_string}.jpg", cropped[1])
        
        return self.start_point