# deepfake-detect
Fake Image Detection Using InceptionResNetV2

This project aims to build a Convolutional Neural Network (CNN) using the InceptionResNetV2 architecture to classify images as real or fake. The dataset contains images categorized into two classes: real and fake. The model is trained on this dataset to achieve accurate classification.

Project Structure
dataset/: Directory containing the dataset with two subdirectories real and fake.
script.py: Python script containing the code for preprocessing, training, and evaluating the model.

README.md: Documentation of the project.

Prerequisites
Ensure you have the following libraries installed:

TensorFlow
NumPy
OpenCV
Matplotlib
Seaborn
Pandas
Scikit-learn

Running the Project
Preprocessing the Data:

The script reads images from the dataset/real and dataset/fake directories.
Images are resized to 256x256 pixels.
Image data is normalized and labels are one-hot encoded.
Splitting the Dataset:

The dataset is split into training and validation sets using an 80-20 split.
Building the Model:

The InceptionResNetV2 model is used as the base model.
A Global Average Pooling layer and a Dense layer with softmax activation are added on top of the base model.
Training the Model:

The model is compiled with binary_crossentropy loss and Adam optimizer.
The model is trained for 10 epochs with a batch size of 20.
Early stopping is used to prevent overfitting.
Evaluating the Model:

After training, the model is evaluated on the validation set.
A confusion matrix is plotted to visualize the performance of the model.
Model Training and Evaluation
The script trains the model and evaluates its performance on the validation set. The confusion matrix is displayed to provide insights into the model's classification performance.



