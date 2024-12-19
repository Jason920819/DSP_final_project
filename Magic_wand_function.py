import numpy as np
import cv2
import time
import os

class MagicWandSketcher:
    def __init__(self, image, mask):
        
        self.image = image.copy()  # 圖像和遮罩的copy，以便後續處理 (會被修改)，是這個東西顯示在GUI panel上面的
        self.original_image = image.copy()  # 保存原始彩色圖像  (永遠不會動當作reference)

        # 將遮罩轉換為二值遮罩：前景(非0、非2值) => 1，其他 => 0
        # 0-背景，1-前景，2-可能的背景，3-可能的前景
        self.mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        self.original_mask = self.mask.copy()  # 用於reset
        self.processed_image = image.copy()   # GrabCut處理後的圖像
        self.processed_mask = self.mask.copy()  # GrabCut處理後的遮罩
        
        self.prev_pt = None  # 記錄上一個滑鼠點位置
        self.brush_size = 5  # 預設畫筆大小
        self.strokes_stack = []

        # initialize GUI
        trackbar_height = 50
        window_height = self.image.shape[0] + trackbar_height
        window_width = self.image.shape[1] * 2  # 兩張圖片並排顯示
        cv2.namedWindow("Magic Wand: Editing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Magic Wand: Editing", window_width, window_height)
        cv2.createTrackbar("Brush Size", "Magic Wand: Editing", 5, 10, self.update_brush_size)

        # mouse function
        cv2.setMouseCallback("Magic Wand: Editing", self.on_mouse)

        # initilize編輯模式
        self.mode = 'foreground'
        self.modes = {
            'foreground': (255, 0, 0),  # 藍色
            'background': (0, 0, 255)   # 紅色
        }

    def update_brush_size(self, val):
        # 畫筆大小限制在1-10內
        self.brush_size = max(1, min(val, 10))

    def show(self):
        # 遮罩去背
        result = self.image * (self.mask != 0)[:, :, np.newaxis]
        # 遮罩轉換為彩色
        mask_color = cv2.cvtColor((self.mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        # 並排顯示pic, mask
        combined = np.hstack([result, mask_color])
        cv2.imshow("Magic Wand: Editing", combined)

    def on_mouse(self, event, x, y, flags, param):
        # 確保只在左側原始圖像上操作
        if x >= self.image.shape[1]:
            return

        pt = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:  # 滑鼠左鍵按下開始
            self.prev_pt = pt
            self.strokes_stack.append((self.mask.copy(), self.image.copy()))

        elif event == cv2.EVENT_LBUTTONUP:  # 放開滑鼠左鍵結束
            self.prev_pt = None

        # 滑鼠左鍵按住時繪製
        if self.prev_pt and flags == cv2.EVENT_FLAG_LBUTTON:
            # 前景為3，背景為0
            mask_val = 3 if self.mode == 'foreground' else 0

            if self.mode == 'foreground':
                # 前景模式：更新遮罩並復原原始圖像
                
                cv2.line(self.mask, self.prev_pt, pt, mask_val, self.brush_size)
                #在mask上面畫線
                ##self.mask = 要更新的mask; prev_pt = 滑鼠起始點; pt = 終點; mask_val = 要更新的值(0 or 3)
            ################
            ##產生彩色圖片的關鍵!當 self.mode 為 foreground 時，在繪製時不僅更新遮罩，還需要將對應的圖像區域用 self.original_image 的像素覆蓋。
                # 創建一個與original mask同樣大小的全零遮罩，在這個臨時遮罩上也繪製同樣的線段，線段值設為 1（表示需要還原的區域），產生彩色圖
                mask_overlay = np.zeros_like(self.mask, dtype=np.uint8)
                cv2.line(mask_overlay, self.prev_pt, pt, 1, self.brush_size)
                # 用 self.original_image 的像素替換 self.image 的像素
                self.image[mask_overlay > 0] = self.original_image[mask_overlay > 0]
            ##############################
            ################
            else:
                # 背景模式：更新遮罩並繪製顏色標記
                color = self.modes[self.mode]
                cv2.line(self.mask, self.prev_pt, pt, mask_val, self.brush_size)
                cv2.line(self.image, self.prev_pt, pt, color, self.brush_size)

            self.prev_pt = pt
            self.show()

    def undo(self):
        if self.strokes_stack:
            # 從stack取出上一個狀態
            self.mask, self.image = self.strokes_stack.pop()
            self.show()
        else:
            print("已是初始狀態，沒有上一步了")

    def run(self):
        while True:
            self.show()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):  # 按F切換到前景模式
                self.mode = 'foreground'
                print("switch to FOREGROUND mode (前景模式)")

            elif key == ord('b'):  # 按B切換到背景模式
                self.mode = 'background'
                print("switch to BACKGROUND mode (背景模式)")

            elif key == ord('r'):  # 重置到GrabCut運算後的狀態
                self.mask = self.processed_mask.copy()
                self.image = self.processed_image.copy()
                self.strokes_stack.clear()
                print("reset successfully! back to GrabCut 運算後的狀態")

            elif key == ord('x'):  # back to last step
                self.undo()

            elif key == 13:  # Enter鍵 = 結束所有操作
                cv2.destroyAllWindows()
                
                # 最終遮罩處理，把設成2跟0的都變成0 (背景)，其他都設成1(前景)。
                # 最後把圖片乘上mask還原出原始的圖案
                final_mask = np.where((self.mask == 2) | (self.mask == 0), 0, 1).astype('uint8')
                result_end = self.original_image * final_mask[:, :, np.newaxis]

                # save results
                cv2.imwrite("./results/After_Magic_wand.png", result_end)
                cv2.imwrite("./results/After_Magic_wand_mask.png", (final_mask * 255).astype('uint8'))
                
                # show results
                cv2.imshow("After_Magic_wand_result!", result_end)
                # press enter to exit
                while True:
                    if cv2.waitKey(1) & 0xFF == 13:
                        break
                print("已儲存結果到 ./results/資料夾")
                break

        cv2.destroyAllWindows()



 
 
 
 



 