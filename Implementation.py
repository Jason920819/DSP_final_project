
import numpy as np
import cv2
import time
import os
from Grabcut_handmade import grabcut, GMM 
from Magic_wand_function import MagicWandSketcher
from Feathering_function import anti_aliasing, show_images_with_gui
from Feathering_function_version2 import feathering_function, show_images_with_gui2
# make sure the directory exists
os.makedirs("./results", exist_ok=True)

# read picture and set the size for 處理圖片(越小越快)
INPUT_IMAGE_PATH = "./raw_picture/26.jpg"
RESIZE_HEIGHT = 400
RESIZE_WIDTH = 400

# 設定grabcut iteration的次數
iter_count = 3 #手刻
iterCount = 3 #builtin function

#設定Feathering的模糊半徑
blur_radius = 3


def load_and_resize_image(image_path, resize_height, resize_width):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"cannot find the file, its path is : {image_path}???")

    height, width = image.shape[:2]
    if height > resize_height or width > resize_width:
        scaling_factor = min(resize_height / height, resize_width / width)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor)
    return image

# read and resize the input picture
image = load_and_resize_image(INPUT_IMAGE_PATH, RESIZE_HEIGHT, RESIZE_WIDTH)
clone = image.copy()
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 一開始預設為全部都是背景 (0)

# 長方形框選
rect = None
rect_drawing = True
# 多邊形框選
points = []
polygon_drawing = True

# 選擇框選模式
mode = int(input("Choose your drawing type!!! (0: 長方形, 1: 多邊形): "))
if mode not in [0, 1]:
    raise ValueError("invalid input，請輸入 0 或 1")

def draw_rectangle(event, x, y, flags, param):
    global rect, rect_drawing
    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵開始框選
        rect = (x, y, x, y)
    elif event == cv2.EVENT_MOUSEMOVE and rect is not None:  # 繪製過程中更新框的右下角座標
        rect = (rect[0], rect[1], x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:  # 右鍵結束框選
        rect = (rect[0], rect[1], x, y)
        rect_drawing = False

def draw_polygon(event, x, y, flags, param):
    global points, polygon_drawing
    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵點擊記錄點
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # 右鍵完成繪製
        polygon_drawing = False

# 繪圖模式選擇 (0: 矩形模式, 1: 多邊形模式)
if mode == 0:  # 長方形模式
    cv2.namedWindow("Draw Rectangle")
    cv2.setMouseCallback("Draw Rectangle", draw_rectangle)

    while rect_drawing:
        temp_image = clone.copy()
        
        # if start drawing
        if rect is not None:
            # show the rect you draw on temp image（綠色，線寬2）
            cv2.rectangle(temp_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        cv2.imshow("Draw Rectangle", temp_image)
        #press right mouse means the end
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    # 紀錄x,y,w,h
    rect = (min(rect[0], rect[2]), 
            min(rect[1], rect[3]), 
            abs(rect[2] - rect[0]), 
            abs(rect[3] - rect[1]))
    
    # show the size of the rectangular
    print(f"選取的矩形框: {rect}")

elif mode == 1:  # 多邊形模式
    cv2.namedWindow("Draw Polygon")
    cv2.setMouseCallback("Draw Polygon", draw_polygon)

    while polygon_drawing:
        temp_image = clone.copy()
        
        # if start drawing
        if len(points) > 1:
            # connect each point
            for i in range(1, len(points)):
                cv2.line(temp_image, points[i - 1], points[i], (0, 255, 0), 2)
            
            # connect first and last point, 變成閉合的
            cv2.line(temp_image, points[-1], points[0], (0, 255, 0), 2)
        
        # 如果只有一個點，draw a circle
        elif len(points) == 1:
            cv2.circle(temp_image, points[0], 5, (0, 255, 0), -1)

        # show the polygon
        cv2.imshow("Draw Polygon", temp_image)
  
        key = cv2.waitKey(1) & 0xFF

        # back to last step（按下 'x' 鍵）
        if key == ord('x') and len(points) > 0:
            points.pop()
        # delete all point and redraw again（按下 'r' 鍵）
        elif key == ord('r'):
            points = []

    cv2.destroyAllWindows()

    # 檢查多邊形頂點數是否足夠（至少3個）
    if len(points) < 3:
        raise ValueError("多邊形頂點數不足，請重新繪製")

    points_np = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, points_np, 3)  # 3 代表 GC_PR_FGD（可能的前景）

# 初始化模型 (前兩個for 手刻function，後兩個for built-in function)
bgd_model = np.zeros((GMM.components_count, 13))  
fgd_model = np.zeros((GMM.components_count, 13))  
bgd_model_builtin = np.zeros((1, 65), dtype=np.float64)
fgd_model_builtin = np.zeros((1, 65), dtype=np.float64)

# 使用自製的 GrabCut
my_mask = mask.copy()
start_time_my = time.time()
if mode == 0:
    grabcut(image, my_mask, rect, bgd_model, fgd_model, iter_count)
else:
    grabcut(image, my_mask, None, bgd_model, fgd_model, iter_count)
end_time_my = time.time()

# 使用 OpenCV 的內建 GrabCut
cv_mask = mask.copy()
start_time_cv = time.time()
if mode == 0:
    cv2.grabCut(image, cv_mask, rect, bgd_model_builtin, fgd_model_builtin, iterCount, mode=cv2.GC_INIT_WITH_RECT)
else:
    cv2.grabCut(image, cv_mask, None, bgd_model_builtin, fgd_model_builtin, iterCount, mode=cv2.GC_INIT_WITH_MASK)
end_time_cv = time.time()
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
print(f"My_GrabCut => {end_time_my - start_time_my:.2f} sec")
print(f"Built_in_GrabCut => {end_time_cv - start_time_cv:.2f} sec")
print("---------------------------------------------------------------")
print("---------------------------------------------------------------")
########################################
########################################
# 將遮罩轉換為二值圖像(0~3變成只有0或1)
def mask_to_binary(mask):
    return np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

my_result = mask_to_binary(my_mask)
my_result_end = image * my_result[:, :, np.newaxis]

cv_result = mask_to_binary(cv_mask)
cv_result_end = image * cv_result[:, :, np.newaxis]



# create 視窗 for顯示結果
def create_comparison_images(original, my_mask, my_result, cv_mask, cv_result):
    # 原始圖像
    cv2.imshow("Original Image", original)

    # 創建 2x2 的結果視窗
    result_image = np.zeros((original.shape[0]*2, original.shape[1]*2, 3), dtype=np.uint8)
    
    # 左上：my_mask
    my_mask_color = cv2.cvtColor((my_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    result_image[:original.shape[0], :original.shape[1]] = my_mask_color
    
    # 右上：built-in mask
    cv_mask_color = cv2.cvtColor((cv_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    result_image[:original.shape[0], original.shape[1]:] = cv_mask_color
    
    # 左下：my_result
    result_image[original.shape[0]:, :original.shape[1]] = my_result
    
    # 右下：built-in result
    result_image[original.shape[0]:, original.shape[1]:] = cv_result
    # 儲存結果圖像
    #cv2.imwrite("./results/my_mask.png", (my_result * 255).astype('uint8'))
    cv2.imwrite("./results/my_result.png", my_result_end)
    #cv2.imwrite("./results/built_in_mask.png", (cv_result * 255).astype('uint8'))
    cv2.imwrite("./results/built_in_result.png", cv_result_end)
    cv2.imwrite("./results/my_mask.png", np.where((my_mask == 1) | (my_mask == 3), 255, 0).astype('uint8'))
    cv2.imwrite("./results/built_in_mask.png", np.where((cv_mask == 1) | (cv_mask == 3), 255, 0).astype('uint8'))
    # show the results
    cv2.putText(result_image, "My Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(result_image, "Built-in Mask", (original.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(result_image, "My Result", (10, original.shape[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(result_image, "Built-in Result", (original.shape[1]+10, original.shape[0]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("GrabCut Results", result_image)
    while True:
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()

create_comparison_images(image, my_result, my_result_end, cv_result, cv_result_end)
# print execution time

""" class MagicWandSketcher:
    def __init__(self, image, mask):
        self.image = image.copy()
        self.mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # GrabCut 後的二值遮罩
        self.original_mask = self.mask.copy()  # 用於完全重置
        self.processed_image = image.copy()   # GrabCut 運算後的圖像
        self.processed_mask = self.mask.copy()  # GrabCut 運算後的遮罩
        self.prev_pt = None
        self.brush_size = 5  # 預設畫筆大小

        # 繪製堆疊（支援撤銷功能）
        self.strokes_stack = []

        # 初始化視窗
        trackbar_height = 50
        window_height = self.image.shape[0] + trackbar_height
        window_width = self.image.shape[1] * 2  # 假設兩張圖片並排
        cv2.namedWindow("Magic Wand: Editing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Magic Wand: Editing", window_width, window_height)
        cv2.createTrackbar("Brush Size", "Magic Wand: Editing", 5, 10, self.update_brush_size)

        cv2.setMouseCallback("Magic Wand: Editing", self.on_mouse)

        # 初始化模式：前景(藍色)和背景(紅色)
        self.mode = 'foreground'
        self.modes = {
            'foreground': (255, 0, 0),  # 藍色
            'background': (0, 0, 255)  # 紅色
        }

    def update_brush_size(self, val):
        self.brush_size = max(1, min(val, 10))  # 限制畫筆大小在 1~10

    def show(self):
        # 計算去背結果
        result = self.image * (self.mask != 0)[:, :, np.newaxis]

        # 創建遮罩的彩色顯示
        mask_color = cv2.cvtColor((self.mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        combined = np.hstack([result, mask_color])

        cv2.imshow("Magic Wand: Editing", combined)

    def on_mouse(self, event, x, y, flags, param):
        if x >= self.image.shape[1]:  # 確保座標不越界
            return

        pt = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:  # 開始繪製
            self.prev_pt = pt
            self.strokes_stack.append((self.mask.copy(), self.image.copy()))  # 儲存當前狀態

        elif event == cv2.EVENT_LBUTTONUP:  # 結束繪製
            self.prev_pt = None

        if self.prev_pt and flags == cv2.EVENT_FLAG_LBUTTON:
            color = self.modes[self.mode]  # 根據模式選擇顏色
            mask_val = 3 if self.mode == 'foreground' else 0  # 遮罩值：前景=3，背景=0

            # 更新遮罩和顯示圖像
            cv2.line(self.mask, self.prev_pt, pt, mask_val, self.brush_size)
            cv2.line(self.image, self.prev_pt, pt, color, self.brush_size)

            self.prev_pt = pt
            self.show()

    def undo(self):
        if self.strokes_stack:
            self.mask, self.image = self.strokes_stack.pop()  # 回復到上一狀態
            self.show()
        else:
            print("無法撤銷，已是初始狀態")

    def run(self):
        while True:
            self.show()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):  # 切換到前景模式
                self.mode = 'foreground'
                print("切換到前景模式")

            elif key == ord('b'):  # 切換到背景模式
                self.mode = 'background'
                print("切換到背景模式")

            elif key == ord('r'):  # 重置到 GrabCut 運算後的狀態
                self.mask = self.processed_mask.copy()
                self.image = self.processed_image.copy()
                self.strokes_stack.clear()
                print("已重置到 GrabCut 運算後的狀態")

            elif key == ord('x'):  # 撤銷上一步操作
                self.undo()

            elif key == 13:  # Enter 鍵儲存結果並退出
                cv2.destroyAllWindows()
                final_mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
                result_end = image * final_mask[:, :, np.newaxis]

                cv2.imwrite("./results/After_Magic_wand.png", result_end)
                cv2.imwrite("./results/After_Magic_wand_mask.png", (final_mask * 255).astype('uint8'))
                cv2.imshow("Final Result", result_end)
                cv2.waitKey(0)
                print("已儲存結果到 ./results/")
                break

        cv2.destroyAllWindows() """


# Use magic wand !!!!          
magic_wand = MagicWandSketcher(image, my_mask)
magic_wand.run()

#feathering !!
feathering_input_path = "./results/After_Magic_wand.png"
feathering_output_path = "./results/After_Feathering.png" 
feathering_input_mask_path = "./results/After_Magic_wand_mask.png"  # 黑白遮罩路徑
#version 1 (黑色衣服會出事被模糊)
#before, after = anti_aliasing(feathering_input_path, feathering_output_path, blur_radius)
#show_images_with_gui(before, after)
#version 2 (解決黑色衣服問題)
before, after = feathering_function(feathering_input_path, feathering_input_mask_path, feathering_output_path, blur_radius)
show_images_with_gui2(before, after)