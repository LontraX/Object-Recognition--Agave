import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import class_weight

# loads the dataset and splits it into train and test sets
def load_dataset(data_dir):
    image_dir = os.path.join(data_dir, 'images') 
    
    annotation_dir = os.path.join(data_dir,'annotations', 'xmls')
    trimap_dir = os.path.join(data_dir,'annotations', 'trimaps') 
    list_file = os.path.join(data_dir,'annotations', 'list.txt') 

    dataset = []
    with open(list_file, 'r') as f:
        lines = f.readlines()[6:]  # Skip the first six lines

        for line in lines:
            line = line.strip().split(' ')
            image_name = line[0]
            class_id = int(line[1]) - 1  # Subtract 1 to make class ID zero-based
            species_id = int(line[2]) - 1  # Subtract 1 to make species ID zero-based
            breed_id = int(line[3]) - 1  # Subtract 1 to make breed ID zero-based

            image_path = os.path.join(image_dir, f"{image_name}.jpg")
            annotation_path = os.path.join(annotation_dir, f"{image_name}.xml")
            trimap_path = os.path.join(trimap_dir, f"{image_name}.png") 

            image, annotation, trimap = preprocess_image(image_path, annotation_path, trimap_path)

            
            dataset.append((image, annotation, trimap, class_id, species_id, breed_id))
    # split the datset into train and test
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    return train_data, test_data


# preprocesses the image, here it resizes the image
def preprocess_image(image_path, annotation_path, trimap_path):
    try:
        # Load image, annotation, and trimap
        image = cv2.imread(image_path)
        annotation = load_annotation(annotation_path)
        trimap = load_trimap(trimap_path)
        # preprocessing steps
        # Resize image to a specific size
        image = cv2.resize(image, (224, 224))
        
        return image, annotation, trimap
    except:
        return None

def load_annotation(annotation_path):
    bounding_boxes = []
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bounding_boxes.append([xmin, ymin, xmax, ymax])
    
        return bounding_boxes
    except:
        return bounding_boxes

def load_trimap(trimap_path):
    try:
        # Load trimap image using OpenCV
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize trimap values to [0, 1]
        trimap = trimap.astype(np.float32) / 255.0
        
        # Threshold trimap to obtain binary mask
        trimap_binary = np.zeros_like(trimap)
        trimap_binary[trimap < 0.25] = 0  # Background
        trimap_binary[trimap > 0.75] = 1  # Foreground
        trimap_binary[(trimap >= 0.25) & (trimap <= 0.75)] = 0.5  # Not classified

        return trimap_binary
    except:
        return None

# this function handles data augmentation on the dataset
def augment_dataset():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,  # Random rotation between -20 and 20 degrees
        width_shift_range=0.1,  # Randomly shift width by 10%
        height_shift_range=0.1,  # Randomly shift height by 10%
        horizontal_flip=True,  # Randomly flip images horizontally
        brightness_range=[0.2,0.8]  # brightness
    )


    return datagen

def evaluate_model(model, test_images, test_labels):
    # Predict the labels for the test images
    predictions = model.predict(test_images)
    
    # Convert the predictions into class labels
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Convert the true labels into class labels
    true_labels = np.argmax(test_labels, axis=1)
    
    # Calculate the accuracy score
    accuracy = np.mean(predicted_labels == true_labels)
    
    return accuracy

def train_model(train_dataset, test_dataset):
    # Load the pre-trained ResNet50 model without the top classification layer
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the pre-trained layers
    base_model.trainable = False

    # Add a new classification layer
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

     # Extract features and labels from the training dataset
    train_images = np.array([item[0] for item in train_dataset])
    train_annotations = np.array([item[1] for item in train_dataset])
    train_labels = np.array([item[3:] for item in train_dataset])

    # Extract features and labels from the test dataset
    test_images = np.array([item[0] for item in test_dataset])
    #test_annotations = np.array([item[1] for item in test_dataset])
    test_labels = np.array([item[3:] for item in test_dataset])

    # Split the training dataset into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    #augment train data
    train_datagen = augment_dataset()
    # Batch size
    batch_size = 32

    model.fit(train_datagen.flow(train_images,train_labels,batch_size=batch_size, shuffle=False),
              epochs=1, validation_data=(val_images, val_labels)
              )
    model.save("detection_model_01.h5")
    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")
    return model

if __name__ == '__main__':
    
    data_dir = 'C:\Users\Olumide Joda\source\repos\object_recognition\Data'
    train_data, test_data = load_dataset(data_dir)
    model = train_model(train_data, test_data)
     # Extract features and labels from the test dataset
    test_images = np.array([item[0] for item in test_data])
    #test_annotations = np.array([item[1] for item in test_data])
    test_labels = np.array([item[3:] for item in test_data])
    # Evaluate the model on the test dataset
    accuracy = evaluate_model(model, test_images, test_labels)
    print(f"Test Accuracy for this model is: {accuracy}")
