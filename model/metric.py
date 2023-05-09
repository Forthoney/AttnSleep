import tensorflow as tf


def accuracy(output, target):
    pred = tf.argmax(output, axis=1)
    pred = tf.expand_dims(pred, axis=1)
    assert pred.shape[0] == len(target)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.float32)).numpy()
    return correct / len(target)


# def confusion_matrix(output, target):
#    pred = tf.argmax(output, axis=1)
#    # pred = tf.expand_dims(pred, axis=1)
#    assert pred.shape[0] == len(target)
#    # target = tf.argmax(target, axis=1)
#    # target = tf.expand_dims(target, axis=1)
#    return confusion_matrix(pred, target)
#
#
# def f1(output, target):
#    pred = tf.argmax(output, axis=1)
#    pred = tf.expand_dims(pred, axis=1)
#    assert pred.shape[0] == len(target)
#    target = tf.argmax(target, axis=1)
#    target = tf.expand_dims(target, axis=1)
#    return f1_score(pred.numpy(), target.numpy(), average="macro")
