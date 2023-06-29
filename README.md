# Object Recognition README

This repository contains code for object recognition using TensorFlow and OpenCV. The code consists of functions for loading and preprocessing the dataset, data augmentation, training the model, and evaluating the model's performance.

## Prerequisites

Make sure you have the following dependencies installed:

- Python (version 3.6 or above)
- TensorFlow (version 2.0 or above)
- OpenCV (version 4.0 or above)
- NumPy (version 1.18 or above)
- scikit-learn (version 0.23 or above)
- xml.etree.ElementTree (built-in library)

## Dataset

The dataset should be organized in the following directory structure:

```
data_dir/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── annotations/
    │   ├── xmls/
    │   │   ├── image1.xml
    │   │   ├── image2.xml
    │   │   └── ...
    │   └── trimaps/
    │       ├── image1.png
    │       ├── image2.png
    │       └── ...
    └── annotations/
        └── list.txt
```

The `images` directory should contain the input images in JPEG format. The `annotations/xmls` directory should contain XML files corresponding to the annotations of the images. The `annotations/trimaps` directory should contain the trimap images in PNG format. The `annotations/list.txt` file should contain a list of image filenames along with their corresponding class, species, and breed IDs.

## Usage

To use the code, follow these steps:

1. Set the `data_dir` variable in the main script to the path of your dataset directory.

2. Run the main script:

```python
python main.py
```

The script will load the dataset, preprocess the images, split the dataset into training and testing sets, train the model, and evaluate its performance.

The model can also be accessed via an API endpoint. 

## Functions


### load_dataset(data_dir)
This function loads the dataset from the specified directory and returns two lists: train_data and test_data. Each item in the list consists of an image, annotation, trimap, class ID, species ID, and breed ID.




### preprocess_image(image_path, annotation_path, trimap_path)
This function takes the paths of an image, its annotation, and trimap, and performs the following preprocessing steps:

Load the image, annotation, and trimap using OpenCV.
Resize the image to a specific size (224x224 pixels).
The function returns the preprocessed image, annotation, and trimap.



### load_annotation(annotation_path)
This function loads the annotation from an XML file and returns a list of bounding boxes. Each bounding box is represented as [xmin, ymin, xmax, ymax].



### load_trimap(trimap_path)
This function loads the trimap image using OpenCV and performs the following steps:

Normalize the trimap values to the range [0, 1].
Threshold the trimap to obtain a binary mask: 0 for background, 1 for foreground, and 0.5 for areas not classified.
The function returns the binary trimap mask.

### augment_dataset()
This function creates an image data generator using TensorFlow's ImageDataGenerator class. It applies various augmentations to the images, such as random rotation, shifting, flipping, and brightness adjustment.

The function returns the data generator.


### evaluate_model(model, test_images, test_labels)
This function takes a trained model, test images, and test labels as input. It predicts the labels for the test images, converts the predictions and true labels into class labels, and calculates the accuracy score.

The function returns the accuracy score.


### train_model(train_dataset, test_dataset)
This function trains the object recognition model using transfer learning with the pre-trained ResNet50 model. It freezes the pre-trained layers, adds a new classification layer on top, compiles the model, and fits it to the training dataset.

The function saves the trained model as a file and returns the model object.

## Techniques and strategies used to handle bias and improve accuracy scores:
- Data preprocessing: The code includes data preprocessing steps such as resizing the images to a specific size (224x224), loading annotations and trimaps, and performing normalization and thresholding on the trimap values. These preprocessing steps help to standardize and enhance the input data for better model performance.

- Data augmentation: The code uses the augment_dataset() function to apply data augmentation techniques such as rotation, width and height shifting, horizontal flipping, and brightness adjustments. Data augmentation helps to increase the diversity and variability of the training data, which can improve the model's ability to generalize to unseen examples and reduce overfitting.

- Transfer learning: The code utilizes transfer learning by loading a pre-trained ResNet50 model and freezing its pre-trained layers. By leveraging the knowledge learned from training on a large dataset (ImageNet), transfer learning allows the model to benefit from the pre-trained weights and extract relevant features from the input images. The model then adds a new classification layer that is trained on the specific task at hand.

- Model architecture: The model architecture consists of a base model (ResNet50) followed by a global average pooling layer, a dense layer with ReLU activation, and a final dense layer with softmax activation for multi-class classification. This architecture is commonly used in transfer learning scenarios and has been shown to be effective for image classification tasks.

- Model training and validation: The code splits the training dataset into training and validation sets using the train_test_split() function. It then uses the fit() function to train the model on the augmented training data, with a specified batch size and a single epoch. The validation data is used to monitor the model's performance during training and prevent overfitting.

- Evaluation: After training the model, the code evaluates its performance on the test dataset using the evaluate() function. It calculates the test loss and accuracy, providing insights into the model's ability to generalize to unseen data.

## Conclusion

This code provides a framework for object recognition using TensorFlow and OpenCV. By following the steps outlined in this README, you can train a model on your own dataset and evaluate its performance. Feel free to modify the code and experiment with different models, architectures, and data augmentation techniques to improve the results.
