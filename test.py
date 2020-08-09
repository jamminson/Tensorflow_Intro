import tensorflow as tf
from tensorflow import keras
import numpy as np
import model_b

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# flattened = list()
# for row in train_images[0]:
#     flattened.extend(row)
#
# print(len(flattened))

# for image in range(train_images.shape[0]):
#     train_images[image].flatten()
#
# print(train_images[0])


# def flatten(np_matrix):
#     flattened = np.zeros(shape=(np_matrix.shape[0], np_matrix.shape[1] ** 2))
#     for image_number in range(np_matrix.shape[0]):
#         arr = list()
#         for row in np_matrix[image_number]:
#             arr.extend(row)
#         arr = np.array(arr)
#         flattened[image_number] = arr
#
#     return flattened
#
#
# a = flatten(train_images)
# print(len(a[0]))
# # print(a[1])

x = np.array([5, 15, 25, 35, 45, 55])
print(len(x))

print(round(5.9345))