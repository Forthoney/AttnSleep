import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score


def accuracy(output: tf.Tensor, target: tf.Tensor):
    """Calculates the model's accuracy

    Args:
        output (tf.Tensor): logis for each class
        target (tf.Tensor): one-hot encoded class labels

    Returns:
        correct / numer of predictions
    """
    pred = tf.argmax(output, axis=1)
    target = tf.argmax(target, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, target), tf.float32))


def confusion(output, target):
    pred = tf.argmax(output, axis=1)
    pred = tf.expand_dims(pred, axis=1)
    target = tf.argmax(target, axis=1)
    target = tf.expand_dims(target, axis=1)
    return confusion_matrix(pred, target)


def f1(output, target):
    pred = tf.argmax(output, axis=1)
    pred = tf.expand_dims(pred, axis=1)
    target = tf.argmax(target, axis=1)
    target = tf.expand_dims(target, axis=1)
    return f1_score(pred.numpy(), target.numpy(), average="macro")
