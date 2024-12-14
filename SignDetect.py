import cv2
import numpy as np
import os

lower_red1 = np.array([0, 70, 50])   
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

input_folder = 'C:/Users/PC/Desktop/stop_sign_dataset' #Klasör yolu
output_folder = 'C:/Users/PC/Desktop/Output' #Output klasörü yolu

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg'):

        image = cv2.imread(os.path.join(input_folder, filename))
        resized_image = cv2.resize(image, (800, 600))

        hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000 or area > 50000:
                continue

            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 8:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                center_x = x + w // 2
                center_y = y + h // 2
                print(f"{filename} dosyasındaki STOP işaretinin merkezi: ({center_x}, {center_y})")

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)

        cv2.imshow(f"Detected STOP Sign - {filename}", resized_image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
