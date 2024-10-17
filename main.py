import cv2
from input_manager import InputManager, HairSEMEvents
from line_tracer import LineTracerManager
from renderer import ImageRenderer
from settings import X_SIZE, Y_SIZE

raw_image_data = cv2.imread("./sem_images/SEM_4.jpg")
resized_image_data = cv2.resize(raw_image_data, (X_SIZE, Y_SIZE))

renderer = ImageRenderer(resized_image_data)
input_manager = InputManager()
line_tracer_manager = LineTracerManager(renderer, input_manager)

cv2.imshow('img_window', resized_image_data)
cv2.setMouseCallback('img_window', input_manager.on_mouse)

while True:
    renderer.reset()
    renderer.tick()
    line_tracer_manager.tick()

    cv2.imshow('img_window', renderer.image)

    event = input_manager.get_keyevent()
    if event == HairSEMEvents.EXIT:
        break
    elif event == HairSEMEvents.REMOVE_PREVIOUS_LINE:
        line_tracer_manager.revert_last()

cv2.destroyAllWindows()