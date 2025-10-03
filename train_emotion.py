import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. Đường dẫn dữ liệu
# =========================
train_dir = "c:/Users/ADMIN/.vscode/train"
test_dir  = "c:/Users/ADMIN/.vscode/test"

IMG_SIZE = (96,96)   # MobileNetV2 yêu cầu ≥96x96
BATCH_SIZE = 64

# =========================
# 2. Data Augmentation
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.85, 1.15)
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",      # đổi sang 3 kênh
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False   # ⚠ rất quan trọng để confusion matrix đúng
)

NUM_CLASSES = train_generator.num_classes
class_labels = list(train_generator.class_indices.keys())

# =========================
# 3. Xây dựng model MobileNetV2
# =========================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96,96,3))

# Freeze gần hết, chỉ fine-tune 20 layer cuối
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# =========================
# 4. Callbacks
# =========================
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")

callbacks = [
    ModelCheckpoint("saved_models/emotion_model.h5", monitor="val_accuracy", save_best_only=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
]

# =========================
# 5. Train
# =========================
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30,
    callbacks=callbacks
)

# =========================
# 6. Vẽ biểu đồ Accuracy & Loss
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()

# =========================
# 7. Đánh giá bằng Confusion Matrix & Classification Report
# =========================
print("\n🔎 Đánh giá trên tập test...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(test_generator.classes, y_pred))

print("\nClassification Report:")
print(classification_report(test_generator.classes, y_pred, target_names=class_labels))

print("✅ Training hoàn tất! Model đã lưu tại: saved_models/emotion_model.h5")
exit()
