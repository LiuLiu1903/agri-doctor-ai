from fastapi import FastAPI, File, UploadFile
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI()

print("⏳ Đang khởi động hệ thống AI...")

VIETNAMESE_NAMES = {
    'Bacterial_Spot': 'Bệnh Đốm Vi Khuẩn',
    'Early_Blight': 'Bệnh Đốm Vòng (Tàn lụi sớm)',
    'Healthy': 'Cây Khỏe Mạnh',
    'Late_Blight': 'Bệnh Sương Mai (Mốc sương)',
    'Leaf_Mold': 'Bệnh Nấm Mốc Lá',
    'Mosaic_Virus': 'Bệnh Khảm Virus',
    'Septoria_Spot': 'Bệnh Đốm Lá Septoria',
    'Spider_Mites': 'Nhện Đỏ',
    'Tomato_Target_Spot': 'Bệnh Đốm Đen',
    'Yellow_Leaf_Curl': 'Bệnh Xoăn Lá Vàng'
}

ALLOWED_KEYWORDS = [
    'plant', 'tree', 'flower', 'leaf', 'grass', 'vegetable', 'fruit', 
    'pot', 'garden', 'broccoli', 'cabbage', 'corn', 'cucumber', 
    'zucchini', 'daisy', 'rose', 'mushroom', 'agriculture'
]

try:
    DISEASE_MODEL = tf.keras.models.load_model("plant_disease_model.h5")
    
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
        class_names_en = {v: k for k, v in class_indices.items()} 
    
    SECURITY_MODEL = MobileNetV2(weights='imagenet')
    
    print(" Hệ thống đã sẵn sàng!")
except Exception as e:
    print(f"     Lỗi khởi động: {e}")
    exit()

def prepare_image(image_bytes, target_size):
    """Đọc và resize ảnh"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    # ==========================================
    # BƯỚC 1: (SECURITY CHECK)
    # ==========================================
    img_sec = prepare_image(image_bytes, (224, 224))
    img_array_sec = img_to_array(img_sec)
    img_array_sec = np.expand_dims(img_array_sec, axis=0)
    img_array_sec = preprocess_input(img_array_sec)

    sec_preds = SECURITY_MODEL.predict(img_array_sec)
    decoded_preds = decode_predictions(sec_preds, top=1)[0]
    top_obj_name = decoded_preds[0][1] 
    top_conf = decoded_preds[0][2]  

    is_plant = any(k in top_obj_name.lower() for k in ALLOWED_KEYWORDS)
    if not is_plant and top_conf > 0.4:
        return {
            "prediction": "Không phải cây",
            "confidence": float(top_conf),
            "message": f"Phát hiện đây là '{top_obj_name}', không phải lá cây."
        }

    # ==========================================
    # BƯỚC 2: (DIAGNOSIS)
    # ==========================================
    img_doc = prepare_image(image_bytes, (224, 224))
    img_array_doc = np.array(img_doc)
    img_array_doc = img_array_doc / 255.0
    img_array_doc = np.expand_dims(img_array_doc, axis=0)

    predictions = DISEASE_MODEL.predict(img_array_doc)
    pred_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    name_en = class_names_en[pred_index]
    name_vi = VIETNAMESE_NAMES.get(name_en, name_en)

    if confidence < 0.50:
        return {
            "prediction": "Không xác định",
            "confidence": confidence,
            "message": "Ảnh không rõ hoặc bệnh chưa được học."
        }

    return {
        "prediction": name_vi,
        "confidence": confidence,
        "original_name": name_en
    }

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8001)