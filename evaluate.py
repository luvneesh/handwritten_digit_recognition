from utils import load_mnist
from keras.utils import np_utils
from vgg16 import VGG16
import matplotlib.pyplot as plt
# from super_learner_extension import SuperLearnerExtension
# import argparse
import numpy as np
import cv2 as cv
import os

PATH = './models/'
models = [VGG16()]

def evaluate(prediction, true_label):
    '''
    Returns the test accuracy.

    Args:
        prediction - 2D numpy array (Number of samples, Class)
        y_test - 2D numpy array (Number of samples, Class)

    Returns:
        Test accuracy (float)
    '''
    pred_indices = np.argmax(prediction, 1)
    true_indices = np.argmax(true_label, 1)

    return np.mean(pred_indices == true_indices)

def main():
    # args = get_argument_parser()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    # print(x_train.shape)
    # plt.matshow(x_train[0].reshape(28, 28), cmap=plt.cm.gray)
    # plt.show()
    print(x_train[0])
    cv.waitKey(0)
    exit()

    predictions = []
    for model in models:
        model_name = type(model).__name__
        model.compile()

        print('Loading pretrained weights for ', model_name, '...', sep='')
        model.load_weights(PATH + model_name + '.h5')

        # In order to save time, stored prediction results can be used.
        prediction_path = './predictions/' + model_name + extension
        if os.path.isfile(prediction_path):
            single_model_prediction = np.load(prediction_path)
            print('Prediction file loaded')
        else:
            print('No prediction file. Predicting...')
            single_model_prediction = model.predict(x, verbose = 1)
            print(single_model_prediction[0])
            np.save(prediction_path, single_model_prediction)
            print('Saved prediction file in', prediction_path)

        predictions.append(single_model_prediction)
        single_model_accuracy = evaluate(single_model_prediction, y)
        print(f'Evaluation of {model_name}:', single_model_accuracy * 100, '%')
        print()

if __name__ == '__main__':
    main()
