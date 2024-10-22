import cv2
from crop_manager import CropManager
from input_manager import InputManager, HairSEMEvents
from line_tracer import LineTracerManager
from renderer import ImageRenderer
from settings import X_SIZE, Y_SIZE

raw_image_data = cv2.imread("./sem_images/images/SEM_D_PRESSURE.jpg")
raw_mask_data = cv2.imread("./sem_images/segmentation-masks/SEM_D_PRESSURE.jpg")

resized_image_data = cv2.resize(raw_image_data, (X_SIZE, Y_SIZE))
resized_mask_data = cv2.resize(raw_mask_data, (X_SIZE, Y_SIZE))

renderer = ImageRenderer(resized_image_data)
mask_renderer = ImageRenderer(resized_mask_data)

input_manager = InputManager()
line_tracer_manager = LineTracerManager(renderer, input_manager)
crop_manager = CropManager(renderer, input_manager)

cv2.imshow('img_window', resized_image_data)
cv2.setMouseCallback('img_window', input_manager.on_mouse)

while True:
    mask_renderer.update()
    # line_tracer_manager.update()
    crop_manager.update()

    cv2.imshow('img_window', mask_renderer.image)

    event = input_manager.get_keyevent()
    if event == HairSEMEvents.EXIT:
        break
    elif event == HairSEMEvents.REMOVE_PREVIOUS_LINE:
        line_tracer_manager.revert_last()
    elif event == HairSEMEvents.SAVE_CROPPED_IMAGE:
        crop_manager.save([renderer.raw_image.copy(), mask_renderer.raw_image.copy()])
        break

cv2.destroyAllWindows()