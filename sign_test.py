import tensorflow as tf
from tensorflow import keras
import model_a_works
import model_b
import numpy as np
import math
from math import comb


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
test_images = test_images / 255.0
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def model_setup(train_x, train_y, test_x):
    model_1 = model_a_works.train_model_a(train_x, train_y)
    model_2 = model_b.train_model_b(train_x, train_y)

    predictions_1 = model_1.predict(test_x)
    predictions_2 = model_2.predict(model_b.flatten(test_x))

    for i in range(len(predictions_1)):
        predictions_1[i] = np.argmax(predictions_1[i])
        predictions_2[i] = int(round(predictions_2[i]))

    return predictions_1, predictions_2


def sign_test_setup(pred_1, pred_2, test_x, test_y):
    null = 0
    plus = 0
    minus = 0

    for image in range(len(test_x)):

        if pred_1[image] != pred_2[image]:
            if pred_2[image] != test_y[image]:
                plus += 1
                null += 1
            else:
                minus += 1
                null += 1

        else:
            if pred_1[image] == test_y[image]:
                null += 1

    return plus, minus, null


def sign_test(q, plus, minus, null):
    p_value = 0
    n = 2 * int(math.ceil(null / 2)) + plus + minus
    k = int(math.ceil(null / 2)) + min(plus, minus)

    for i in range(k + 1):
        p_value += (comb(n, i) * (q ** i) * ((1 - q) ** (n - i)))
        # p_value += (comb(n, i) * (q ** n))

    return round((2 * p_value) * 100, 1)


(model_1, model_2) = model_setup(train_images, train_labels, test_images)
(plus, minus, null) = sign_test_setup(model_1, model_2, test_images, test_labels)
print(sign_test(0.5, plus, minus, null))
