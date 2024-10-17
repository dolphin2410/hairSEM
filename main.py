import cv2
from events import HairSEMEvents
from input_manager import InputManager
from line_tracer import LineTracer, LineTracerManager
from cv_layers import ImageRenderer
from settings import X_SIZE, Y_SIZE

raw_image_data = cv2.imread("./sem_images/SEM_4.jpg")

# reversed_image_data = cv2.bitwise_not(raw_image_data)
reversed_image_data = cv2.resize(raw_image_data, (X_SIZE, Y_SIZE))

renderer = ImageRenderer(reversed_image_data)
input_manager = InputManager()
line_tracer_manager = LineTracerManager(renderer, input_manager)

cv2.imshow('img_window', reversed_image_data)
cv2.setMouseCallback('img_window', input_manager.on_mouse)

while True:
    renderer.tick()
    line_tracer_manager.tick()

    cv2.imshow('img_window', renderer.image)
    
    renderer.reset()

    event = input_manager.get_keyevent()

    if event == HairSEMEvents.EXIT:
        break
    if event == HairSEMEvents.REMOVE_PREVIOUS_LINE: # move this line
        line_tracer_manager.revert_last()

cv2.destroyAllWindows()