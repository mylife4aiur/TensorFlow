# Introduction

A high-level API like ```tf.contrib.learn``` helps you manage data sets, estimators, training and inference.

名称中带有contrib的函数仍处于开发状态
Note that a few of the high-level TensorFlow APIs--those whose method names contain contrib-- are still in development


# Computational Graph
A tensor's rank is its number of dimensions.

You might think of TensorFlow Core programs as consisting of two discrete sections:
@@ -31,6 +32,7 @@ TensorFlow中的操作都是按照Computational Graph来的，Graph中的最小�

> To actually evaluate the nodes, we must run the computational graph within a session. A session encapsulates the control and state of the TensorFlow runtime.

## placeholders
placeholders是一个能接受外部参数的node，不同于constant，可以传入数据。

第一个参数是所执行的模型，可以是一个数或list...，参数数值的输入是字典的形式，返回一个和fetch相同形状(shape)的对象
@@ -81,6 +83,89 @@ run(fetches, feed_dict=None, options=None, run_metadata=None)

> In machine learning we will typically want a model that can take arbitrary inputs, such as the one above. To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:

## Loss Function

A loss function measures how far apart the current model is from the provided data

```python
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    #Init is just a HANDLE that initializes all the global variables.
    init = tf.global_variables_initializer()
    # Until we call sess.run, the variables are uninitialized.
    sess.run(init)
    # define a placeholder node
    y = tf.placeholder(tf.float32)
    # vector square
    squared_deltas = tf.square(linear_model - y)
    # sum all vector entries into a scalar
    loss = tf.reduce_sum(squared_deltas)
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

## variable reassign

A variable is initialized to the value provided to tf.Variable but can be changed using operations like tf.assign

```python
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    #Init is just a HANDLE that initializes all the global variables.
    init = tf.global_variables_initializer()
    # Until we call sess.run, the variables are uninitialized.
    sess.run(init)
    # define a placeholder node
    y = tf.placeholder(tf.float32)
    # vector square
    squared_deltas = tf.square(linear_model - y)
    # sum all vector entries into a scalar
    loss = tf.reduce_sum(squared_deltas)

    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    # Noticed that session.run need to be call to use the node
    sess.run([fixW, fixb])
    print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```



## 总结

新node的使用必须要initialized(除constant)，具体的为调用```sess = tf.Session() ```和```sess.run(node)```。




# Train A Model Using tf.train API










# TensorBoard
TensorFlow provides a utility called TensorBoard that can display a picture of the computational graph. Here is a screenshot showing how TensorBoard visualizes the graph:
\ No newline at end of file