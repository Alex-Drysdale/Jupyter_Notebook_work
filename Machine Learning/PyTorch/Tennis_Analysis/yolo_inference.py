from ultralytics import YOLO

model = YOLO('yolov8x') # x for extra large

result = model.predict('Input_videos/image.png')

print(result)
print("boxes")
for box in result[0].boxes:
    print(box)