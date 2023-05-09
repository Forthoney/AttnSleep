import tensorflow as tf


def weighted_CrossEntropyLoss(output, target):
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return ce(target, output)
