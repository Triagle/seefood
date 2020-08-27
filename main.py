#!/usr/bin/env python3

import tensorflow as tf
from PIL import Image
import model

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image", help="image to classify")
    args = parser.parse_args()

    cnn = tf.keras.models.load_model(model.MODEL_PATH)
    print(cnn)
    img = Image.open(args.image)
    img = img.resize((model.IMG_WIDTH, model.IMG_HEIGHT), Image.BILINEAR)
    tensor = tf.keras.preprocessing.image.img_to_array(img)
    tensor = tf.reshape(tensor, (1,) + tensor.shape)
    output = tf.nn.sigmoid(cnn.predict(tensor))
    print("not_dog" if output[0][0] > 0.5 else "hot_dog")


if __name__ == "__main__":
    main()
