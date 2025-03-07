import tensorflow as tf
import os
from src.preprocess import train_ds, test_ds, class_names

# Image dimensions
img_height = 224
img_width = 224
num_classes = len(class_names)

# Load Pretrained MobileNetV2 Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# 🔥 Freeze More Layers to Prevent Overfitting (Fine-tune only last 10 layers)
for layer in base_model.layers[:-60]:
    layer.trainable = False

# 🔁 Data Augmentation (Improves Generalization)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.5),  # More randomness
    tf.keras.layers.RandomZoom(0.5),
    tf.keras.layers.RandomContrast(0.6),
    tf.keras.layers.RandomTranslation(height_factor=0.4, width_factor=0.4),
    tf.keras.layers.RandomBrightness(0.2)  # Adding brightness variation
])

# Apply augmentation only during dataset loading
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Define Model
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.6),  # 🔥 Increased dropout to 60%
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 📉 Learning Rate Schedule (Reduces LR over time)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-6,  # Lower starting LR for fine-tuning
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile Model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 🔥 Save only the BEST model using checkpointing
checkpoint_path = "best_model.keras"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'
)

# 🔁 Reduce Learning Rate When Accuracy Stalls
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,  # Reduce LR by 50%
    patience=20,  # If no improvement in 20 epochs
    min_lr=1e-7  # Minimum allowed LR
)

# 🚀 Adjust Early Stopping (Patience 100 instead of 400)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=100,  # Stop if no improvement for 100 epochs
    restore_best_weights=True
)

# 🔄 Resume Training From Best Model If Exists
if os.path.exists(checkpoint_path):
    print("🔄 Resuming training from best saved model...")
    model = tf.keras.models.load_model(checkpoint_path)

# 🚀 Train the model for 10,000 epochs
epochs = 10000
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=epochs,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr]
)

# ✅ Load the best saved model after training
model = tf.keras.models.load_model(checkpoint_path)

# Evaluate performance
test_loss, test_acc = model.evaluate(test_ds)
print(f"🔥 Final Fine-Tuned Test Accuracy: {test_acc * 100:.2f}%")

# Save final model manually (just in case)
model.save("asl_translator_best.keras")
print("✅ Final model saved as asl_translator_best.keras")
