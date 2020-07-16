import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Sequential


def splitting_video_to_frames(video_file, dir_path):
    count = 0
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(5)  # frame rate
    while cap.isOpened():
        frame_id = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if ret is False:
            break

        if frame_id % math.floor(frame_rate) == 0:
            filename = 'frame%d.jpg' % count
            count += 1
            cv2.imwrite(f'{dir_path}/{filename}', frame)
    cap.release()
    return


def map_frames_with_code(map_data, path_dir):
    """
    0 – neither JERRY nor TOM
    1 – for JERRY
    2 – for TOM
    """
    res = [plt.imread(f'{path_dir}/{img_name}') for img_name in map_data.Image_ID]
    return np.array(res)


def reshape_and_validate_data(data_array, add_value=False):
    images = []
    output_shape = (224, 224, 3) if add_value else (224, 224)
    for step in range(0, data_array.shape[0]):
        res = resize(data_array[step], preserve_range=True, output_shape=output_shape)  # reshaping to 224*224*3
        images.append(res.astype(int))
    return preprocess_input(np.array(images))


def make_base_model(base_model, train):
    train = base_model.predict(train).reshape(208, 7 * 7 * 512)
    return train / train.max()


def build_model():
    # Building the model
    model = Sequential()
    model.add(InputLayer((7 * 7 * 512,)))  # input layer
    model.add(Dense(units=1024, activation='sigmoid'))  # hidden layer
    model.add(Dense(3, activation='softmax'))  # output layer

    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
