import cv2
from ultralytics.yolov5 import YOLOv5

model = YOLOv5('yolov5s.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame, size=640)

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,  255,  0),  2)
        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) -  10),
                    cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0,  255,  0),  2)

    cv2.imshow('YOLOv5 Object Detection', frame)

    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
