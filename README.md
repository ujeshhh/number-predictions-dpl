# deeplearning number predicition using tensorflow

In this project, I have built a machine learning model using TensorFlow and Keras to classify handwritten digits from the MNIST dataset, which consists of 28x28 pixel images of digits 0-9. The goal is to develop a model that can correctly predict the digit in an image based on its pixel values.

### Steps Involved:

1. **Loading the Data:**
   I used the MNIST dataset, which is readily available in TensorFlow. It contains a training set and a test set:
   - `x_train` and `y_train` are the images and labels for the training data.
   - `x_test` and `y_test` are the images and labels for testing the model's accuracy.

2. **Data Normalization:**
   I normalized the pixel values of the images so that each pixel's value is between 0 and 1. This helps the neural network learn more effectively.

3. **Building the Model:**
   - I created a neural network with a `Sequential` model, which is a linear stack of layers.
   - The first layer is a `Flatten` layer that reshapes the 28x28 images into a 1D array, making it suitable for input into the dense layers.
   - The next two layers are `Dense` layers with 128 neurons each, activated using ReLU (Rectified Linear Unit) function. These layers learn the complex relationships between the pixel values and the corresponding digit.
   - The final layer is a `Dense` layer with 10 neurons, representing the 10 digits (0-9). It uses the `softmax` activation function to output a probability distribution over the digits.

4. **Compiling the Model:**
   - I used the `Adam` optimizer, which is an adaptive learning rate optimization algorithm.
   - The loss function used is `sparse_categorical_crossentropy`, suitable for multi-class classification problems.
   - I also included `accuracy` as a metric to track the model's performance.

5. **Training the Model:**
   - I trained the model for 3 epochs using the training data. The model learned to classify the handwritten digits by minimizing the loss function.

6. **Evaluating the Model:**
   - After training, I evaluated the model using the test data (`x_test` and `y_test`). The accuracy of the model on the test data is around 96.4%, indicating that the model generalizes well to new, unseen data.

7. **Saving and Loading the Model:**
   - Once the model was trained, I saved it to a file (`epic_num_reader.keras`) using the `model.save()` method.
   - I also demonstrated how to load the saved model using `tf.keras.models.load_model()`, allowing the model to be used for future predictions.

8. **Prediction:**
   - I made predictions using the trained model on the test set. The model outputs probabilities for each of the 10 digits, with the highest value corresponding to the predicted digit.

This project demonstrates the power of neural networks in solving real-world problems like digit recognition. It involves data preprocessing, building a neural network model, training the model, and making predictions. The model is able to classify handwritten digits with high accuracy, which is useful in various applications like automated number reading, postal services, and more.
