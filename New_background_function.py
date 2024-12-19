""" #############################
# 不是class型的
#############################
import cv2
import numpy as np
import os

# Thumbnail size for background selection
THUMBNAIL_WIDTH = 400
THUMBNAIL_HEIGHT = 220
# size of background when editing (加上去背的圖案的背景大小) 設800 = 800x800
Ideal_background_size = 800

def resize_image(image, max_size, upscale=False):

    h, w = image.shape[:2]
    if max(h, w) > max_size or (upscale and max(h, w) < max_size):
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

def select_background():

    while True:
        category = input("Select category: 0 for human, 1 for animal: ")
        if category in ['0', '1']:
            break
        print("Invalid input. Please enter 0 or 1.")

    folder = "./Backgrounds/human" if category == '0' else "./Backgrounds/animals"
    images = []

    # Load and resize images
    for i in range(1, 7):
        path = os.path.join(folder, f"{i}.jpg")
        img = cv2.imread(path)
        if img is not None:
            # Scale each image to fit within the defined thumbnail size
            scale = min(THUMBNAIL_WIDTH / img.shape[1], THUMBNAIL_HEIGHT / img.shape[0])
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            images.append((img, path))

    if not images:
        print("No images found in the selected category.")
        return None

    # Dynamically calculate grid size
    cols = 3
    rows = -(-len(images) // cols)  # Ceiling division
    grid_width = cols * THUMBNAIL_WIDTH
    grid_height = rows * THUMBNAIL_HEIGHT

    # Create the grid
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, (img, path) in enumerate(images):
        img_h, img_w = img.shape[:2]
        y_offset = (idx // cols) * THUMBNAIL_HEIGHT
        x_offset = (idx % cols) * THUMBNAIL_WIDTH

        y_start = max(0, y_offset + (THUMBNAIL_HEIGHT - img_h) // 2)
        x_start = max(0, x_offset + (THUMBNAIL_WIDTH - img_w) // 2)

        # Ensure range is valid
        if y_start + img_h > grid_height or x_start + img_w > grid_width:
            print(f"Image {path} exceeds grid bounds, skipping.")
            continue

        try:
            grid[y_start:y_start + img_h, x_start:x_start + img_w] = img
        except ValueError as e:
            print(f"Error inserting image {path}: {e}")
            continue

    # Dynamically resize window
    cv2.namedWindow("Select Background", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Background", grid_width, grid_height)

    def mouse_click(event, x, y, flags, param):
        nonlocal selected_path
        if event == cv2.EVENT_LBUTTONDOWN:
            col = x // THUMBNAIL_WIDTH
            row = y // THUMBNAIL_HEIGHT
            idx = row * cols + col
            if 0 <= idx < len(images):
                selected_path = images[idx][1]

    selected_path = None
    cv2.setMouseCallback("Select Background", mouse_click)

    while True:
        cv2.imshow("Select Background", grid)
        key = cv2.waitKey(1)
        if selected_path:
            break

    cv2.destroyAllWindows()
    return selected_path

def overlay_image(background, overlay, x, y, scale, angle, flip):

    h, w = overlay.shape[:2]
    resized_overlay = cv2.resize(overlay, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Apply rotation
    center = (resized_overlay.shape[1] // 2, resized_overlay.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_overlay = cv2.warpAffine(resized_overlay, rotation_matrix, (resized_overlay.shape[1], resized_overlay.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # Apply flipping
    if flip:
        rotated_overlay = cv2.flip(rotated_overlay, 1)

    overlay_h, overlay_w = rotated_overlay.shape[:2]

    # Constrain overlay position within background boundaries
    x = max(0, min(x, background.shape[1] - overlay_w))
    y = max(0, min(y, background.shape[0] - overlay_h))

    # Define the region of interest on the background image
    roi = background[y:y + overlay_h, x:x + overlay_w]

    # Ensure the overlay fits within the background dimensions
    if roi.shape[:2] != rotated_overlay.shape[:2]:
        return background

    # Separate the alpha channel (if exists)
    if rotated_overlay.shape[2] == 4:
        overlay_rgb = rotated_overlay[:, :, :3]
        alpha = rotated_overlay[:, :, 3] / 255.0
    else:
        overlay_rgb = rotated_overlay
        alpha = np.ones((overlay_h, overlay_w))

    # Blend the overlay with the region of interest
    for c in range(0, 3):
        roi[:, :, c] = (alpha * overlay_rgb[:, :, c] + (1 - alpha) * roi[:, :, c])

    # Put the blended ROI back into the background image
    background[y:y + overlay_h, x:x + overlay_w] = roi

    return background

def main():
    # Select background image
    background_path = select_background()
    if not background_path:
        print("No background selected.")
        return

    overlay_path = "./results/built_in_result.png"
    background = cv2.imread(background_path)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

    if background is None or overlay is None:
        print("Error: Could not load one or both images.")
        return

    # Resize background to fit within 500x500 while maintaining aspect ratio, allow upscaling
    background = resize_image(background, Ideal_background_size, upscale=True)

    # Initial position and scale
    init_x = background.shape[1] // 2 - overlay.shape[1] // 2
    init_y = background.shape[0] // 2 - overlay.shape[0] // 2
    init_scale = 1.0
    init_angle = 0
    init_flip = False

    objects = [{'x': init_x, 'y': init_y, 'scale': init_scale, 'angle': init_angle, 'flip': init_flip}]

    def update_image(event, xx, yy, flags, param):
        nonlocal objects

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, obj in enumerate(objects):
                obj_w = int(overlay.shape[1] * obj['scale'])
                obj_h = int(overlay.shape[0] * obj['scale'])
                if obj['x'] <= xx <= obj['x'] + obj_w and obj['y'] <= yy <= obj['y'] + obj_h:
                    param['dragging'] = i
                    param['prev_x'], param['prev_y'] = xx, yy
                    break

        elif event == cv2.EVENT_MOUSEMOVE and param['dragging'] is not None:
            dx, dy = xx - param['prev_x'], yy - param['prev_y']
            obj = objects[param['dragging']]
            obj['x'] = max(0, min(obj['x'] + dx, background.shape[1] - int(overlay.shape[1] * obj['scale'])))
            obj['y'] = max(0, min(obj['y'] + dy, background.shape[0] - int(overlay.shape[0] * obj['scale'])))
            param['prev_x'], param['prev_y'] = xx, yy

        elif event == cv2.EVENT_LBUTTONUP:
            param['dragging'] = None

    # Create a window for editing
    cv2.namedWindow("Edit Image")
    params = {'dragging': None, 'prev_x': 0, 'prev_y': 0}
    cv2.setMouseCallback("Edit Image", update_image, params)

    while True:
        temp_image = background.copy()
        for obj in objects:
            temp_image = overlay_image(temp_image, overlay, obj['x'], obj['y'], obj['scale'], obj['angle'], obj['flip'])

        cv2.imshow("Edit Image", temp_image)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        elif key == 13:  # ENTER key to finalize
            final_image = temp_image.copy()
            break
        elif key == ord('w'):  # Increase scale of selected object
            if params['dragging'] is not None:
                new_scale = objects[params['dragging']]['scale'] + 0.1
                # Ensure the new scale does not exceed background dimensions
                obj_w = int(overlay.shape[1] * new_scale)
                obj_h = int(overlay.shape[0] * new_scale)
                if obj_w <= background.shape[1] and obj_h <= background.shape[0]:
                    objects[params['dragging']]['scale'] = new_scale
        elif key == ord('s'):  # Decrease scale of selected object
            if params['dragging'] is not None:
                objects[params['dragging']]['scale'] = max(0.1, objects[params['dragging']]['scale'] - 0.1)
        elif key == ord('a'):  # Rotate left
            if params['dragging'] is not None:
                objects[params['dragging']]['angle'] -= 10
        elif key == ord('d'):  # Rotate right
            if params['dragging'] is not None:
                objects[params['dragging']]['angle'] += 10
        elif key == ord('q'):  # Toggle flip
            if params['dragging'] is not None:
                objects[params['dragging']]['flip'] = not objects[params['dragging']]['flip']
        elif key == ord('r'):  # Reset
            objects = [{'x': init_x, 'y': init_y, 'scale': init_scale, 'angle': init_angle, 'flip': init_flip}]
        elif key == ord('c'):  # Add a new object
            if len(objects) < 5:
                objects.append({'x': init_x, 'y': init_y, 'scale': init_scale, 'angle': init_angle, 'flip': init_flip})
        elif key == ord('d'):  # Delete selected object
            if params['dragging'] is not None:
                del objects[params['dragging']]
                params['dragging'] = None

    cv2.destroyAllWindows()

    # Save the final image
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "new_background.png")
    cv2.imwrite(result_path, final_image)

    # Display the result
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() """


#############################################
#class類型
#############################
import cv2
import numpy as np
import os

# Thumbnail size for background selection
THUMBNAIL_WIDTH = 400
THUMBNAIL_HEIGHT = 220
# Size of background when editing
IDEAL_BACKGROUND_SIZE = 800

def resize_image(image, max_size, upscale=False):
    h, w = image.shape[:2]
    if max(h, w) > max_size or (upscale and max(h, w) < max_size):
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image
    return image

def select_background():
    while True:
        category = input("Select category: 0 for human, 1 for animal: ")
        if category in ['0', '1']:
            break
        print("Invalid input. Please enter 0 or 1.")

    folder = "./Backgrounds/human" if category == '0' else "./Backgrounds/animals"

    # Get all image files from the folder
    all_images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
    if not all_images:
        print("No images found in the selected category.")
        return None

    # Randomly select six images
    selected_images = all_images[:6]

    images = []
    for img_name in selected_images:
        path = os.path.join(folder, img_name)
        img = cv2.imread(path)
        if img is not None:
            # Scale each image to fit within the defined thumbnail size
            scale = min(THUMBNAIL_WIDTH / img.shape[1], THUMBNAIL_HEIGHT / img.shape[0])
            new_width = int(img.shape[1] * scale)
            new_height = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            images.append((img, path))

    if not images:
        print("No images found in the selected category.")
        return None

    # Dynamically calculate grid size
    cols = 3
    rows = -(-len(images) // cols)  # Ceiling division
    grid_width = cols * THUMBNAIL_WIDTH
    grid_height = rows * THUMBNAIL_HEIGHT

    # Create the grid
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    for idx, (img, path) in enumerate(images):
        img_h, img_w = img.shape[:2]
        y_offset = (idx // cols) * THUMBNAIL_HEIGHT
        x_offset = (idx % cols) * THUMBNAIL_WIDTH

        y_start = max(0, y_offset + (THUMBNAIL_HEIGHT - img_h) // 2)
        x_start = max(0, x_offset + (THUMBNAIL_WIDTH - img_w) // 2)

        grid[y_start:y_start + img_h, x_start:x_start + img_w] = img

    # Dynamically resize window
    cv2.namedWindow("Select Background", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Background", grid_width, grid_height)

    def mouse_click(event, x, y, flags, param):
        nonlocal selected_path
        if event == cv2.EVENT_LBUTTONDOWN:
            col = x // THUMBNAIL_WIDTH
            row = y // THUMBNAIL_HEIGHT
            idx = row * cols + col
            if 0 <= idx < len(images):
                selected_path = images[idx][1]

    selected_path = None
    cv2.setMouseCallback("Select Background", mouse_click)

    while True:
        cv2.imshow("Select Background", grid)
        key = cv2.waitKey(1)
        if selected_path:
            break

    cv2.destroyAllWindows()
    return selected_path

def overlay_image(background, overlay, x, y, scale, angle, flip):
    # 提取 alpha 通道并裁剪到非透明区域
    if overlay.shape[2] == 4:  # 确保 alpha 通道存在
        alpha_channel = overlay[:, :, 3]
        coords = cv2.findNonZero(alpha_channel)
        x_min, y_min, w, h = cv2.boundingRect(coords)
        overlay_cropped = overlay[y_min:y_min+h, x_min:x_min+w]
    else:
        overlay_cropped = overlay

    # 调整大小
    h, w = overlay_cropped.shape[:2]
    resized_overlay = cv2.resize(overlay_cropped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 旋转
    h, w = resized_overlay.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 计算旋转后的大小
    abs_cos = abs(np.cos(np.radians(angle)))
    abs_sin = abs(np.sin(np.radians(angle)))
    new_w = int(w * abs_cos + h * abs_sin)
    new_h = int(w * abs_sin + h * abs_cos)
    
    rotation_matrix[0, 2] += (new_w - w) / 2
    rotation_matrix[1, 2] += (new_h - h) / 2
    
    rotated_overlay = cv2.warpAffine(resized_overlay, rotation_matrix, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # 翻转
    if flip:
        rotated_overlay = cv2.flip(rotated_overlay, 1)

    # 严格限制在背景范围内
    overlay_h, overlay_w = rotated_overlay.shape[:2]
    max_x = background.shape[1] - overlay_w
    max_y = background.shape[0] - overlay_h
    
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))

    # 处理 ROI
    roi = background[y:y+overlay_h, x:x+overlay_w]
    
    # 混合图像
    if rotated_overlay.shape[2] == 4:  # 有 alpha 通道
        overlay_rgb = rotated_overlay[:, :, :3]
        alpha = rotated_overlay[:, :, 3] / 255.0
    else:
        overlay_rgb = rotated_overlay
        alpha = np.ones((overlay_h, overlay_w))
    
    # 确保 alpha 和 overlay_rgb 的大小与 roi 匹配
    alpha = alpha[:roi.shape[0], :roi.shape[1]]
    overlay_rgb = overlay_rgb[:roi.shape[0], :roi.shape[1]]
    
    for c in range(3):
        roi[:, :, c] = (alpha * overlay_rgb[:, :, c] + (1 - alpha) * roi[:, :, c])
    
    background[y:y+overlay_h, x:x+overlay_w] = roi
    return background

def main():
    # Select background image
    background_path = select_background()
    if not background_path:
        print("No background selected.")
        return

    overlay_path = "./results/After_Feathering.png"
    background = cv2.imread(background_path)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel if available

    if background is None or overlay is None:
        print("Error: Could not load one or both images.")
        return

    # Resize background to fit within IDEAL_BACKGROUND_SIZE while maintaining aspect ratio
    background = resize_image(background, IDEAL_BACKGROUND_SIZE, upscale=True)

    # Initial position and scale
    init_x = background.shape[1] // 2 - overlay.shape[1] // 2
    init_y = background.shape[0] // 2 - overlay.shape[0] // 2
    init_scale = 1.0
    init_angle = 0
    init_flip = False

    objects = [{'x': init_x, 'y': init_y, 'scale': init_scale, 'angle': init_angle, 'flip': init_flip}]

    def update_image(event, xx, yy, flags, param):
        nonlocal objects

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, obj in enumerate(objects):
                obj_w = int(overlay.shape[1] * obj['scale'])
                obj_h = int(overlay.shape[0] * obj['scale'])
                if obj['x'] <= xx <= obj['x'] + obj_w and obj['y'] <= yy <= obj['y'] + obj_h:
                    param['dragging'] = i
                    param['prev_x'], param['prev_y'] = xx, yy
                    break

        elif event == cv2.EVENT_MOUSEMOVE and param['dragging'] is not None:
            dx, dy = xx - param['prev_x'], yy - param['prev_y']
            obj = objects[param['dragging']]
            obj['x'] += dx
            obj['y'] += dy
            param['prev_x'], param['prev_y'] = xx, yy

        elif event == cv2.EVENT_LBUTTONUP:
            param['dragging'] = None

    # Create a window for editing
    cv2.namedWindow("Edit Image")
    params = {'dragging': None, 'prev_x': 0, 'prev_y': 0}
    cv2.setMouseCallback("Edit Image", update_image, params)

    while True:
        temp_image = background.copy()
        for obj in objects:
            temp_image = overlay_image(temp_image, overlay, obj['x'], obj['y'], obj['scale'], obj['angle'], obj['flip'])

        cv2.imshow("Edit Image", temp_image)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        elif key == 13:  # ENTER key to finalize
            final_image = temp_image.copy()
            break
        elif key == ord('w'):  # Increase scale of selected object
            if params['dragging'] is not None:
                objects[params['dragging']]['scale'] += 0.1
        elif key == ord('s'):  # Decrease scale of selected object
            if params['dragging'] is not None:
                objects[params['dragging']]['scale'] = max(0.1, objects[params['dragging']]['scale'] - 0.1)
        elif key == ord('a'):  # Rotate left
            if params['dragging'] is not None:
                objects[params['dragging']]['angle'] -= 10
        elif key == ord('d'):  # Rotate right
            if params['dragging'] is not None:
                objects[params['dragging']]['angle'] += 10
        elif key == ord('q'):  # Toggle flip
            if params['dragging'] is not None:
                objects[params['dragging']]['flip'] = not objects[params['dragging']]['flip']
        elif key == ord('r'):  # Reset
            objects = [{'x': init_x, 'y': init_y, 'scale': init_scale, 'angle': init_angle, 'flip': init_flip}]
        elif key == ord('c'):  # Add a new object
            if len(objects) < 5:
                objects.append({'x': init_x, 'y': init_y, 'scale': init_scale, 'angle': init_angle, 'flip': init_flip})
        elif key == ord('e'):  # Delete selected object
            if params['dragging'] is not None:
                del objects[params['dragging']]
                params['dragging'] = None

    cv2.destroyAllWindows()

    # Save the final image
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "new_background.png")
    cv2.imwrite(result_path, final_image)

    # Display the result
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
