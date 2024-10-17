import cv2
from input_manager import InputManager
from line_tracer import LineTracer
from cv_layers import ImageRenderer

raw_image_data = cv2.imread("./sem_images/SEM_4.jpg")

reversed_image_data = cv2.bitwise_not(raw_image_data)
reversed_image_data = cv2.resize(reversed_image_data, (1000, 600))

renderer = ImageRenderer(reversed_image_data)
input_manager = InputManager()

old_line_tracers = []
line_tracer = LineTracer(renderer, input_manager, None)

cv2.imshow('img_window', reversed_image_data)
cv2.setMouseCallback('img_window', input_manager.on_mouse)

while True:
    renderer.tick()

    for old_tracer in old_line_tracers:
        old_tracer.tick()

    line_tracer.tick()

    cv2.imshow('img_window', renderer.image)
    
    renderer.reset()

    if line_tracer.lock:
        old_line_tracers.append(line_tracer)
        line_tracer = LineTracer(renderer, input_manager, None)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    
    if key & 0xFF == 27:
        old_line_tracers.pop()

cv2.destroyAllWindows()