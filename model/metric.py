import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score

def accuracy(output, target):
    pred = tf.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    correct = tf.reduce_sum(tf.cast(tf.equal(pred, target), tf.float32)).numpy()
    return correct / len(target)

def f1(output, target):
    pred = tf.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return f1_score(pred.numpy(), target.numpy(), average='macro')
