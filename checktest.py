import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print_tensors_in_checkpoint_file(file_name='./saved_uns_model/model.ckpt', tensor_name='iteration', all_tensors=False)
print_tensors_in_checkpoint_file(file_name='./saved_uns_model/model.ckpt', tensor_name='', all_tensors=True)

iteration = tf.Variable(tf.zeros([]), name='iteration')

saver = tf.train.Saver()
sess = tf.Session()

saver.restore(sess, './saved_uns_model/model.ckpt')
print(sess.run(iteration))