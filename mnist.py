import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('D:/deep_learning/MNIST_data',one_hot=True)

#add one more layer and return the outputs of this layer
def add_layer(inputs, in_size, out_size, n_layer, activate_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("Weight"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name+'/Weights',Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1)
            tf.summary.histogram(layer_name+'/biases',biases)
        Wx_plus_b = tf.matmul(inputs, Weights)+biases
    if activate_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activate_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs,ys:v_ys})
    return result

#define placeholder for input to Network
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None,784])    #rows = 28*28 cols defined by number of the input images
    ys = tf.placeholder(tf.float32,[None,10])     #Numer 0~9
 
#add output layer
prediction = add_layer(xs, 784, 10, n_layer=1, activate_function = tf.nn.softmax)
#the error between the prediction and real data
with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("D:/deep_learning/logs",tf.get_default_graph())
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        result = sess.run(merged,feed_dict={xs:batch_xs,ys:batch_ys})
        writer.add_summary(result,i)