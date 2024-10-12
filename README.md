# RK3588_YOLOV8_INFER
优雅的RK3588模型推理方案,使用pybind11绑定c++
```
使用方式：
import sgai_yolo
import cv2
infer = sgai_yolo.Yolo("../yolov8s.float.rknn")

image = cv2.imread("../bus.jpg")
results = infer.commit(image)
for item in results:
    name,score,x,y,w,h = str(item.className),float(item.confidence),item.box.x,item.box.y,item.box.width,item.box.height
    print(name,score,x,y,w,h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 在框上方显示信息
    info = f'{name} ({score})'
    (text_width, text_height), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(image, info, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
cv2.imwrite('output_image.jpg', image)
```

