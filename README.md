The chosen data set is the Digit Recognizer from the provided data set. The data contain a greyscale handwritten digits from 0-9. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The goal is to train a simple model to recognize the handwritten digits. The chosen model at the moment is MLPClassifier (Multi-layer Perceptron) from scikit-learn. This is because the task of classifying 28x28 image are not the most complex. Even a simpler model like MLP can perform well. Later we might change to ResNet or other CNN-based architecture

