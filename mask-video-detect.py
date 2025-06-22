import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_map = {
    0: 'without_mask',
    1: 'with_mask',
    2: 'mask_weared_incorrect'
}

model = torch.load('model/mobilenet/mask-model-new.pth', weights_only=False)
model.eval()

# Load OpenCV DNN Face Detector
face_net = cv2.dnn.readNetFromCaffe(
    'model/face-detection/deploy.prototxt',
    'model/face-detection/res10_300x300_ssd_iter_140000.caffemodel'
)


cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_img)
            face_pil.crop((x1, y1, x2, y2))
            face_pil.show
            input_tensor = preprocess(face_pil).unsqueeze(0)

            # Run model
            with torch.no_grad():
                output = model(input_tensor)
                label_idx = torch.argmax(output, dim=1).item()
                label = mask_map[label_idx]

            # Draw box and label
            color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            

    cv2.imshow('frame1', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cam.release()

cv2.destroyAllWindows()
