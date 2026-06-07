import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import deque

MODELS_DIR     = "models"
NUM_CLASSES    = 5
IMG_SIZE       = 224
CONF_THRESHOLD = 0.75
SMOOTH_FRAMES  = 10

CLASS_NAMES = {
    0: ("glass",   "玻璃瓶",   (255, 165,   0)),
    1: ("metal",   "金屬類",   (0,   200, 255)),
    2: ("paper",   "紙類",     (0,   220,   0)),
    3: ("plastic", "塑膠類",   (80,  120, 255)),
    4: ("trash",   "其他垃圾", (160, 160, 160)),
}

try:
    font_large = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 36)
    font_small = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 22)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(f"{MODELS_DIR}/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

prob_buffer = deque(maxlen=SMOOTH_FRAMES)

cap = cv2.VideoCapture(0)
print("鏡頭啟動，按 Q 結束")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    box_size = int(min(h, w) * 0.6)
    cx, cy   = w // 2, h // 2
    x1, y1   = cx - box_size // 2, cy - box_size // 2
    x2, y2   = cx + box_size // 2, cy + box_size // 2

    roi    = frame[y1:y2, x1:x2]
    rgb    = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    prob_buffer.append(probs)
    avg_probs = np.mean(prob_buffer, axis=0)
    pred      = int(np.argmax(avg_probs))
    conf      = float(avg_probs[pred]) * 100

    if conf >= CONF_THRESHOLD * 100:
        _, _, color   = CLASS_NAMES[pred]
        box_color     = tuple(reversed(color))
    else:
        box_color = (200, 200, 200)

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(img_pil)
    draw.rectangle([(0, 0), (w, 85)], fill=(30, 30, 30))

    if conf >= CONF_THRESHOLD * 100:
        en_name, zh_name, color = CLASS_NAMES[pred]
        label     = f"{zh_name}  ({en_name})"
        conf_text = f"信心分數：{conf:.1f}%"
    else:
        label     = "請將垃圾放入框內"
        conf_text = f"信心分數：{conf:.1f}%  (偵測中...)"
        color     = (200, 200, 200)

    draw.text((15, 8),  label,     font=font_large, fill=color)
    draw.text((15, 55), conf_text, font=font_small,  fill=(200, 200, 200))

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Garbage Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
