import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16

from train_model import (
    splitting_video_to_frames,
    map_frames_with_code,
    reshape_and_validate_data,
    build_model,
)

if __name__ == '__main__':
    splitting_video_to_frames('Tom_and_Jerry.mp4', 'frames')
    splitting_video_to_frames('Tom_and_Jerry3.mp4', 'test_frames')

    data = pd.read_csv('mapping.csv')
    test = pd.read_csv('testing.csv')

    mapped_array = map_frames_with_code(data, 'frames')
    test_array = map_frames_with_code(test, 'test_frames')

    dummy_data = np_utils.to_categorical(data.Class)
    train_frames = reshape_and_validate_data(mapped_array)
    test_frames = reshape_and_validate_data(test_array, True)

    X_train, X_valid, y_train, y_valid = train_test_split(train_frames, dummy_data, test_size=0.3, random_state=42)

    base_model = VGG16(
        weights='imagenet',
        include_top=False,  # to remove the top layer
        input_shape=(224, 224, 3),
    )

    X_train = base_model.predict(X_train).reshape(208, 7 * 7 * 512)
    X_valid = base_model.predict(X_valid).reshape(90, 7 * 7 * 512)
    test_image = base_model.predict(test_frames).reshape(186, 7 * 7 * 512)

    # zero centered images
    centered_train = X_train / X_train.max()
    centered_valid = X_valid / X_train.max()
    centered_test_image = test_image / test_image.max()

    model = build_model()
    model.fit(centered_train, y_train, epochs=50, validation_data=(centered_valid, y_valid))

    predictions = model.predict_classes(centered_test_image)
    print('The screen time of JERRY is', predictions[predictions == 1].shape[0], 'seconds')
    print('The screen time of TOM is', predictions[predictions == 2].shape[0], 'seconds')
