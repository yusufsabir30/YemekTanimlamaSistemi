# Library
import cv2
import numpy as np
from ultralytics import YOLO

# Function
def detect_food(image, model_path):

    
    green_color = (0,255,0)
    blue_color = (0,0,255) # BGR -- RGB
    font = cv2.FONT_HERSHEY_SIMPLEX

    print(" Fotoğraf yükleniyor !")
    image_array = np.asarray(image).copy()
    
    print(" İşlem başladı !")
    model = YOLO(model_path)
    results = model(image_array)[0]
    

    is_detected = len(results.boxes.data.tolist())
    
    if is_detected is not 0:
        threshold = 0.3
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if score > threshold:
                

                cv2.rectangle(image_array, (x1 + 8 ,y1 + 8), (x2 + 8 ,y2 + 8 ), green_color, 2)
                
                score = score * 100
                class_name = results.names[class_id]
                
                text = f"{class_name}: %{score:.2f}"
                cv2.putText(image_array, text, (x1,y1-10), font, 1, green_color, 2, cv2.LINE_AA)

    else:
        text = "No Detection"
        
       
        cv2.putText(image_array, text, (10,30), font, 1, blue_color, 2, cv2.LINE_AA)

    return image_array, is_detected
