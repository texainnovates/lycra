import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Define the directories for training and validation data
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

# Define ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

validation_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    preprocessing_function=preprocess_input
)

# Create the data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(244, 244),
    batch_size=1,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(244, 244),
    batch_size=1,
    class_mode='categorical'
)

# Load the pre-trained EfficientNetB7 model
base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(244, 244, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(2, activation='softmax')(x)  # 2 classes: good and bad

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Stage 1: Feature Extraction - Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model for feature extraction
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for feature extraction
print("Stage 1: Feature Extraction")
model.fit(
    train_generator,
    epochs=10,  # First stage: fewer epochs for feature extraction
    validation_data=validation_generator
)

# Stage 2: Fine-Tuning - Unfreeze some base model layers
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers for fine-tuning
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model for fine-tuning
print("Stage 2: Fine-Tuning")
model.fit(
    train_generator,
    epochs=30,  # Second stage: more epochs for fine-tuning
    validation_data=validation_generator
)

# Save the entire model to an .h5 file
model.save('efficientnetb7_model_finetuned(NEW).h5')
print("Model saved to 'efficientnetb7_model_finetuned(NEW).h5'")
