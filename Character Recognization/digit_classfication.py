import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),#reformat the data(two dimensions to one dimensions. i.o.w unstack rows and lining up data in a single line)
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',#for updating weights w.r.t loss functions and data
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),#to steer the model in the right direction
              metrics=['accuracy'])#to monitor the training and testing steps
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)#A prediction is an array of 10 numbers. They represent the model's "confidence"...
# that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value:
img = test_images[1]
img = (np.expand_dims(img,0))
predictions_single = probability_model.predict(img)
result_index = np.argmax(predictions_single)#get the max value from predictions array
print('The images is ',result_index)
plt.imshow(test_images[1],cmap=plt.cm.binary)
plt.show()
# plt.figure(figsize=(10,10))
# plt.imshow(train_images[0],cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()          