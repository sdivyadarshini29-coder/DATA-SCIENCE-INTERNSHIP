import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ==========================================
# 1. LOAD AND PREPROCESS BRAIN IMAGES
# ==========================================
print("🔄 Loading Dataset and Preprocessing...")

# Normalize pixels (0-255 to 0.0-1.0) and set 20% data for validation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training Dataset
train_generator = datagen.flow_from_directory(
    'Brain_Tumor_Detection',   # Folder name
    target_size=(128, 128),     # Resizing to 128x128
    batch_size=32,
    class_mode='binary',        # Yes or No split
    subset='training',
    classes=['no', 'yes']       # 'pred' folder-ah train panna koodadhu
)

# Validation Dataset
validation_generator = datagen.flow_from_directory(
    'Brain_Tumor_Detection',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    classes=['no', 'yes']
)

# ==========================================
# 2. BUILDING THE CNN ARCHITECTURE
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),           
    layers.Dense(1, activation='sigmoid') 
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 3. START CNN MODEL TRAINING
# ==========================================
print("\n--- 🚀 STARTING REAL BRAIN TUMOR TRAINING ---")
model.fit(
    train_generator,
    epochs=5,                      # 5 rounds/epochs
    validation_data=validation_generator
)

# Save the trained weights model file
model.save('brain_tumor_cnn_model.h5')
print("\n✅ Saved Successfully as 'brain_tumor_cnn_model.h5'!")


# ==========================================
# 4. AUTO-TESTING WITHOUT CV2 (PURE TERMINAL TEXT OUTPUT)
# ==========================================
print("\n--- 📸 RUNNING AUTO-TEST ON A PREDICTION IMAGE ---")

pred_folder = os.path.join('Brain_Tumor_Detection', 'pred')
if os.path.exists(pred_folder) and len(os.listdir(pred_folder)) > 0:
    # Pred folder-la irukra muthal image-ah auto-ah pick panrom
    test_img_name = os.listdir(pred_folder)[0]
    test_img_path = os.path.join(pred_folder, test_img_name)
    
    print(f"Testing image path: {test_img_path}")
    
    # Pure image preprocessing using Keras helper wrapper (No CV2 needed)
    img = image.load_img(test_img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = x / 255.0                  # Normalization
    x = np.expand_dims(x, axis=0)
    
    # Predict using model
    prediction = model.predict(x)
    
    # Clean output box formatting on terminal screen
    print("\n==========================================")
    print("           AI PREDICTION RESULT           ")
    print("==========================================")
    if prediction[0][0] > 0.5:
        print("🔴 Result: BRAIN TUMOR DETECTED! ⚠️")
    else:
        print("🟢 Result: HEALTHY BRAIN SCAN (NO TUMOR) ✅")
    print(f"Confidence score: {prediction[0][0]:.4f}")
    print("==========================================\n")
else:
    print("ℹ️ 'pred' folder empty-ah iruku, testing skip seiyapadugiradhu.")