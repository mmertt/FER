from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Define the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Directory paths to your dataset
train_data_dir = 'dataset'

# Function to perform data augmentation to create 3000 images for a class if *_augmented directory doesn't exist
def augment_class(class_name):
    print(class_name)
    # Set up a directory for the augmented images
    output_dir = f'{train_data_dir}/train_aug/{class_name}'
    
    # Check if the *_augmented directory already exists
    if not os.path.exists(output_dir):
        # Create the directory if it doesn't exist
        os.makedirs(output_dir)
        
        # Select images belonging to the specific class
        class_images = os.listdir(f'{train_data_dir}/train/{class_name}')

        # Shuffle the images to randomize selection
        random.shuffle(class_images)

        # Ensure exactly 3000 augmented images are created for each class
        images_needed = 2000
        images_per_original = (images_needed + len(class_images) - 1) // len(class_images)

        total_images_created = 0
        for img_name in class_images:
            if total_images_created >= images_needed:
                break

            img_path = f'{train_data_dir}/train/{class_name}/{img_name}'
            img = load_img(img_path, color_mode='rgb', target_size=(48, 48))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=f'{class_name}_aug_{i}', save_format='jpg'):
                i += 1
                total_images_created += 1
                print(total_images_created)
                if i >= images_per_original:
                    break

# Apply augmentation to each class that has fewer than 3000 images or select 3000 if more than 3000 exist
classes_to_augment = ['angry', 'disgust', 'fear', 'neutral', 'sad', 'surprise', 'happy']

for class_name in classes_to_augment:
    augment_class(class_name)


# Check if GPU is available and visible to TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("Num GPUs Available: ", len(physical_devices))
else:
    print("No compatible GPU available or TensorFlow cannot access GPU.")


train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255)

train_set = train_datagen.flow_from_directory(
    train_data_dir + "/train_aug",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    classes=None, subset="training")

val_set = train_datagen.flow_from_directory(
    train_data_dir + "/train_aug",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    classes=None, subset="validation")

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    train_data_dir + "/test",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode='categorical')

def Classes_Count( path, name):
    Classes_Dict = {}

    for Class in os.listdir(path):

        Full_Path = os.path.join(path, Class)
        Classes_Dict[Class] = len(os.listdir(Full_Path))

    df = pd.DataFrame(Classes_Dict, index=[name])

    return df

Train_Count = Classes_Count("dataset/train_aug", 'Train').transpose().sort_values(by="Train", ascending=False)
Test_Count = Classes_Count("dataset/test", 'Test').transpose().sort_values(by="Test", ascending=False)

df = pd.concat([Train_Count,Test_Count] , axis=1)


# Define the model
cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[48, 48, 1]),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(7, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

# Train the model
cnn.fit(
    x=train_set,
    validation_data=val_set,
    epochs=20
)

CNN_Score = cnn.evaluate(test_set)

print("    Test Loss: {:.5f}".format(CNN_Score[0]))
print("Test Accuracy: {:.2f}%".format(CNN_Score[1] * 100))

CNN_Predictions = cnn.predict(test_set)

# Choosing highest probalbilty class in every prediction
CNN_Predictions = np.argmax(CNN_Predictions, axis=1)

cm=confusion_matrix(test_set.labels, CNN_Predictions)

print(cm)
