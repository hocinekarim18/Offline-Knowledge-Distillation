# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from resnet import resnet_layer, resnet_v1
from Distiller import Distiller_AdaIn




print("================ Data Loading ================")
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Data shapes
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("")

print("Building strategy")
teacher = resnet_v1(input_shape=(32, 32, 3), depth= 26)
teacher.summary()



print("")
BATCH_SIZE = 32
EPOCHS = [150, 100, 70, 30]
lr = 1



print("================ Training model ================")
tf.random.set_seed(10)
for epoch in EPOCHS:
  lr = lr/10 
  teacher.compile(
      optimizer=tf.keras.optimizers.SGD(learning_rate= lr),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  )

  # Train and evaluate on data.
  hist = teacher.fit(x_train, y_train, epochs=epoch, batch_size = BATCH_SIZE)

  teacher.evaluate(x_test, y_test)
  print("")
  
  
  print("Saving model ")
  teacher.save("trained_resnet26")
  print("Saving Done !")
  
print("End !")
  
  
  
