import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

cel = np.array( [-40, -10, 0, 10, 30], dtype='float' )
fah = np.array( [-40, 14, 32, 50, 86], dtype='float' )

# Step1 Define a layer
l0 = tf.keras.layers.Dense( units=1, input_shape=[1] )
l1 = tf.keras.layers.Dense( units=5)
l2 = tf.keras.layers.Dense( units=1)

# Step 2 Form model from layers
model = tf.keras.models.Sequential([
    l0,
    l1,
    l2
])

# Step 3: compile and train model 
# Learning rate 
model.compile( loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.1) )

# output 
hist = model.fit( cel, fah, epochs = 2000, verbose= False )

print("Done training the model")
print(l0.get_weights() )
print(l1.get_weights() )
print(l2.get_weights() )


plt.xlabel('epochs')
plt.ylabel('error')
plt.plot( hist.history['loss'] )
plt.show()

print(model.predict([100, 120]))

