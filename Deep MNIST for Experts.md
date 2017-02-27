<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
#   Beginners

MNIST is a simple computer vision dataset. It consists of images of handwritten digits

What we will accomplish in this tutorial:

*   Learn about the MNIST data and softmax regressions
*   Create a function that is a model for recognizing digits, based on looking at every pixel in the image
*   Use TensorFlow to train the model to recognize digits by having it "look" at thousands of examples (and run our first TensorFlow session to do so)
*   Check the model's accuracy with our test data

##  MNIST data

The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation)

每张照片可以用一个大array表示，最终将矩阵flatten成一个向量，怎么flatten不重要，重要的是每张图片的处理保持一致。
> We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't matter how we flatten the array, as long as we're consistent between images. 

> Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.

是否采取flatten操作取决于选用的算法是否需要利用到2D的信息。

Our labels is "one-hot vectors". A one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension.

输入的数据是一个[55000, 784]的tensor，因为将一个矩阵变成了长度为784
的向量，标签值是一个[55000, 10]的tensor，因为采用了"one-hot vectors"

> mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784], mnist.train.labels is a [55000, 10] array of floats.

##  Softmax Regressions
Softmax gives us a list of values between 0 and 1 that add up to 1. Even later on, when we train more sophisticated models, the final step will be a layer of softmax

A softmax regression has two steps: 
*   add up the evidence of our input being in certain classes
*   convert that evidence into probabilities.

softmax : exponentiating its inputs and then normalizing them

softmax是一种归一化的处理方法，将向量中的0，负数映射为约等于0，呈倍数的放大正数的比例，同时减小一部分的比例

![](‪softmax.jpg)

> The exponentiation means that one more unit of evidence increases the weight given to any hypothesis multiplnicatively. And conversely, having one less unit of evidence means that a hypothesis gets a fraction of its earlier weight. No hypothesis ever has zero or negative weight.

SVM只选自己喜欢的男神，Softmax把所有备胎全部拉出来评分，最后还归一化一下


##  Implement

### Build Regression Model
在python做高效率的数值运算，经常使用Numpy这样的库，使用其他语言在python外部实现诸如矩阵乘积的昂贵操作。但是在将计算结果 switching back to Python的时候会存在overhead。这一现象在使用GPU或分布式的时候尤为明显，因为会有a high cost to transferring data。

TensorFlow的解决方案仍然是 does its heavy lifting outside Python，但是采用了计算图(computation graph)以避免overhead，不同于以往每一个 expensive operation 都要从python中独立出来。将原来的运算关系变为了计算图中的interacting operations


```python
    import tensorflow as tf

    # 定义x是一个可传入参数的变量(placeholder)，传入参数的shape为[None, 784]，其中None代表任意长度
    x = tf.placeholder(tf.float32, [None, 784])

    # Variable是一个可改变的tensor，一般需要确定的参数都设为Variable
    # Variable不一定都需要在定义的时候就被初始化
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # model
    # 使用matmul求矩阵乘积 
    y = tf.nn.softmax(tf.matmul(x, W) + b)

```

### Training
The cost or the loss represents how far off our model is from our desired outcome. 

One very common, very nice function to determine the loss of a model is called "cross-entropy."

$$H_{y'}(y)=-\sum_i{y_i'log(y_i)}$$


Where y is our predicted probability distribution, and y′ is the true distribution (the one-hot vector with the digit labels).

```python
    y_ = tf.placeholder(tf.float32, [None, 10])

    # tf.log后维度不变
    # 这里的*对应matlab中的.* elementwise product
    # reduce_sum, sum the elements ALONG the reduction_indices (从0开始计数)
    # tf.reduce_mean computes the mean over all the examples in the batch.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

然而，最终版并没有使用这个loss函数，因为数值不稳定( numerically unstable)，而vsoftmax_cross_entropy_with_logits 在内部计算了激活函数softmax而获得了更好的数值稳定性。


```python
    # 这里取均值是为了和之后的batch相对应
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

而由于将整个计算图都传给了TensorFlow，所以能自动实现参数的求解, 通过使用梯度下降，来最小化cross entropy来确定参数

当然TensorFlow还有许多其他的[优化函数](https://www.tensorflow.org/api_guides/python/train#optimizers)

```python
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

至此整个模型已搭建完毕，下面需要把这个模型launch到一个sesion中
```python
    sess = tf.InteractiveSession()
```

初始化定义的变量
```python
    tf.global_variables_initializer().run()
```

迭代求解
```python
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

feed_dict应该是这个计算图所需要的所有参数的字典，这样在每次循环中都能更新一次batch

Each step of the loop, we get a "batch" of one hundred random data points from our training set. We run train_step feeding in the batches data to replace the placeholders.

### Evaluating Our Model

tf.argmax 给出最大的entry的index，along第二个参数所对应的axis，主义者已编号是从0开始的，因此0对应行，1对应列

tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis. For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label. We can use tf.equal to check if our prediction matches the truth.

```python
    # 返回一个boolean，判断二者是否对应相等
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # tf.cast将bool值转化成float值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```


### 总结

完整代码
```python
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data

    # 导入数据集，这里不全，前面省略了一部分导入数据的代码
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # 定义模型输入的特征值x
    x = tf.placeholder(tf.float32, [None, 784])
    # 定义模型参数
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # 定义SOFTMAX regression model
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 定义Label值
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 计算Loss函数cross_entropy
    # tf.reduce_mean computes the mean over all the examples in the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # 定义指定训练方法的优化器(Optimizer)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # 定义一个Session
    sess = tf.InteractiveSession()
    # 初始化变量
    tf.global_variables_initializer().run()

    # 训练1000次
    for _ in range(1000):
          batch_xs, batch_ys = mnist.train.next_batch(100)
          sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    # Evaluation
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # tf.cast将bool值转化成float值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

#   Experts

##  Session

TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.

InteractiveSession class 
允许

> It allows you to interleave operations which build a computation graph with ones that run the graph. This is particularly convenient when working in interactive contexts like IPython. If you are not using an InteractiveSession, then you should build the entire computation graph before starting a session and launching the graph.




$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)


#   Deep
In this tutorial we will learn the basic building blocks of a TensorFlow model while constructing a deep convolutional MNIST classifier.