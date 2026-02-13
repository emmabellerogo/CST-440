import tensorflow as tf

print(tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
