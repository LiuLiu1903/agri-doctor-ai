import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import json


IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 
DATA_DIR = 'dataset/train'

if not os.path.exists(DATA_DIR):
    print(f"❌ LỖI: Không tìm thấy thư mục '{DATA_DIR}'. Hãy kiểm tra lại!")
    exit()

print("🔄 Đang load dữ liệu và tạo Augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,     
    rotation_range=20,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,   
    validation_split=0.2 
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

labels = train_generator.class_indices
print(f" Tìm thấy các bệnh: {labels}")
with open('class_indices.json', 'w') as f:
    json.dump(labels, f)
print(" Đã lưu danh sách bệnh vào 'class_indices.json'")

# --- 2. XÂY DỰNG MODEL (TRANSFER LEARNING) ---
print("🏗️ Đang tải MobileNetV2 (Pre-trained)...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. TIẾN HÀNH TRAIN ---
print("🚀 BẮT ĐẦU TRAIN MODEL...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# --- 4. LƯU KẾT QUẢ ---
print(" Đang lưu model...")
model.save('plant_disease_model.h5')
print(" THÀNH CÔNG! Model đã được lưu tại 'plant_disease_model.h5'")