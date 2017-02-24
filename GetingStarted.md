
# e
A high-level API like ```tf.contrib.learn``` helps you manage data sets, estimators, training and inference.


Note that a few of the high-level TensorFlow APIs--those whose method names contain contrib-- are still in development

# Tensors
A tensor's rank is its number of dimensions.

You might think of TensorFlow Core programs as consisting of two discrete sections:

*   Building the computational graph.
*   Running the computational graph.

TensorFlow中的操作都是按照Computational Graph来的，Graph中的最小单位是Node，通过Session执行，需要注意的是变量值只有在执行后才会显示，在building的过程中无法返回数值，只能返回一个对象。


```python
    # Constant
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(node1, node2) # object, can't output a value
    sess = tf.Session()
    print(sess.run([node1, node2]))
```

> A computational graph is a series of TensorFlow operations arranged into a graph of nodes

> Notice that printing the nodes does not output the values as you might expect. Instead, they are nodes that, when evaluated, would produce 3.0 and 4.0, respectively. 

> To actually evaluate the nodes, we must run the computational graph within a session. A session encapsulates the control and state of the TensorFlow runtime.

placeholders是一个能接受外部参数的node，不同于constant，可以传入数据。

第一个参数是所执行的模型，可以是一个数或list...，参数数值的输入是字典的形式，返回一个和fetch相同形状(shape)的对象

```python
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b  # + provides a shortcut for tf.add(a, b)

    print(sess.run(adder_node, {a: 3, b:4.5}))
    print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
```

run(fetches, feed_dict=None, options=None, run_metadata=None)

*   fetches: A single graph element, a list of graph elements, or a dictionary whose values are graph elements or lists of graph elements (described above).
*   feed_dict: A dictionary that maps graph elements to values (described above).
*   options: A [RunOptions] protocol buffer
*   run_metadata: A [RunMetadata] protocol buffer

> A graph can be paramaterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later.

## Variable和Constant的区别

为了使模型可训练，我们需要让模型对相同的数据得到新的输出。Variable使我们向图中加入可训练的参数。
不同于Costant在定义时就有值，Variable需要init才能使用

```python
    # Constant
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0) # also tf.float32 implicitly
    print(node1, node2)
    sess = tf.Session()
    print(sess.run([node1, node2]))

    #Variable
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    #Init is just a HANDLE that initializes all the global variables.
    init = tf.global_variables_initializer()
    # Until we call sess.run, the variables are uninitialized.
    sess.run(init)

    print(sess.run(linear_model, {x:[1,2,3,4]}))
```

> In machine learning we will typically want a model that can take arbitrary inputs, such as the one above. To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:


# TensorBoard
TensorFlow provides a utility called TensorBoard that can display a picture of the computational graph. Here is a screenshot showing how TensorBoard visualizes the graph: