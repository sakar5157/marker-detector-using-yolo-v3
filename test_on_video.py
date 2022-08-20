import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#loading Deep Neural Network for YOLO weights and yolo config file with classes name specified and 
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_training.cfg")
classes = ['Marker']
layer_names = net.getLayerNames()
#print(layer_names)
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
#print(output_layers)


while True:
    ret,img = cap.read()
    img = cv2.resize(img, None, fx=0.6, fy=0.6)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.10:
                obj_name = classes[class_id] # Get obj name classes list with classes id index
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img = cv2.putText(img, obj_name, (x-5, y-5), font, 1.5, (255,0,0), 2)
        cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()