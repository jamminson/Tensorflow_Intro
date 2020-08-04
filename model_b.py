import numpy as np
from sklearn.linear_model import LinearRegression


def train_model_b(images, labels):
    images = flatten(images)
    print(images.shape)
    model = LinearRegression().fit(images, labels)
    return model
    # r_sq = model.score(x_train, y_train)
    # y_pred = model.predict(x)
    # print(y_pred)


def flatten(np_matrix):
    flattened = np.zeros(shape=(np_matrix.shape[0], np_matrix.shape[1]**2))
    for image_number in range(np_matrix.shape[0]):
        arr = list()
        for row in np_matrix[image_number]:
            arr.extend(row)

        arr = np.array(arr)
        flattened[image_number] = arr

    return flattened
