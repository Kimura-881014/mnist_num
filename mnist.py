import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps, ImageFilter

mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train),(x_test, y_test) = mnist
x_train = x_train / 255.0
x_test = x_test / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input((28, 28)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128))
model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation(tf.keras.activations.softmax))

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

model.fit(x_train, y_train, epochs=1)
print(model.evaluate(x_test, y_test))

# 予測関数
def pre_num(num):
    img = Image.open(str(num) + '.jpg').convert('L')
    img.thumbnail((28,28)) 
    img = ImageOps.invert(img)
    img = img.filter(filter=ImageFilter.BLUR)
    img = np.array(img) / 255.0

    pred = model.predict(img[np.newaxis])
    return np.argmax(pred)

for i in range(10):
    print(pre_num(i))
