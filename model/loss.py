import tensorflow as tf

def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    ce = tf.CategoricalCrossentropy(from_logits=True, weight=tf.constant(classes_weights, dtype=tf.float32))
    return ce(target, output)