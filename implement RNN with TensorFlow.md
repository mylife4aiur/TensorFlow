# 使用TensorFlow实现RNN     Edit by Haoyu

## A Noob's Guide
[原文链接](http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/)

### Task

统计长度为20的二进制数中1的个数

### 数据生成

```python
    import numpy as np
    from random import shuffle
    # 生成前2**20个20位二进制
    train_input = ['{0:020b}'.format(i) for i in range(2**20)]
    # 随机打乱
    shuffle(train_input)
    # 将字符串转化为包含int元素的list
    train_input = [map(int,i) for i in train_input]

    # 将得到的list处理成input形式
    ti  = []
    for i in train_input:
        temp_list = []
        for j in i:
                temp_list.append([j])
        ti.append(np.array(temp_list))
    train_input = ti

```

TensorFlow 要求输入数据是一个tensor，维度为 [batch_size, sequence_length, input_dimension]， sequence_length可以理解为时间序列，input_dimension为某一个时间点上所输入数据的维度。这个task下sequence_length=20，input_dimension=1

map(function, iterable, ...)
Apply function to every item of iterable and return a list of the result

### 设计模型
LSTM cell 的定义所涉及到的参数

在定义cell的时候需要定义隐藏层参数
The value of it is it up to you, too high a value may lead to overfitting or a very low value may yield extremely poor results. 

state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state
这里官方文档没有说明白， c_state代表cell state，m_state代表hidden state

True:error 0.1%

As many experts have put it, selecting the right parameters is more of an art than science.

computation graph...

RNN VS. DYNAMIC_RNN
简单地说，just use tf.nn.dynamic_rnn
RNN，先建立，后执行，更慢，且无法传入超过规定长度的序列
DYNAMIC_RNN 边执行，边动态建立模型

> Internally, tf.nn.rnn creates an unrolled graph for a fixed RNN length. That means, if you call tf.nn.rnn with inputs having 200 time steps you are creating a static graph with 200 RNN steps. First, graph creation is slow. Second, you’re unable to pass in longer sequences (> 200) than you’ve originally specified.

> tf.nn.dynamic_rnn solves this. It uses a tf.While loop to dynamically construct the graph when it is executed. That means graph creation is faster and you can feed batches of variable size. What about performance? You may think the static rnn is faster than its dynamic counterpart because it pre-builds the graph. In my experience that’s not the case.

## 官网教程




[参考文章](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)

## 使用 tf.train.SequenceExample() 进行数据处理
[参考文章一](http://feisky.xyz/machine-learning/rnn/sequence.html)

基本的，一个Example中包含Features，Features里包含Feature（这里没s）的字典。最后，Feature里包含有一个 FloatList， 或者ByteList，或者Int64List

推荐使用tf.SequenceExample载入数据，而不是直接用python中的array直接导入。  使用tf.SequenceExample的好处是：

*   方便分布式训练，Tensorflow 内置分布式学习模块
*   便于数据模型复用，使用者只需要将数据以SequenceExample方式导入即可
*   便于使用tensorflow内置的其他函数，比如 ```tf.parse_single_sequence_example```
*   分离数据和模型。



数据预处理的全过程大致可分为：

*   数据格式转化 : Convert your data into tf.SequenceExample format
*   将数据写入TFRecord文件 : Write one or more TFRecord files with the serialized data
*   使用函数读取数据文件 : Use tf.TFRecordReader to read examples from the file
*   使用分析函数对数据作分析 : Parse each example using tf.parse_single_sequence_example

#### parse_single_sequence_example

这个函数将一个时间序列example解析为一个映射到tensor的字典的tuple

> This op parses a serialize sequence example into a tuple of dictionaries mapping keys to Tensor and SparseTensor objects respectively.

```python
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=ex,
        context_features=context_features,
        sequence_features=sequence_features
    )
```




第一个返回值(字典类型)包含 ```context_features``` 中键的映射，第二个返回值(字典类型)包含 ```sequence_features``` 中键的映射。

> The first dictionary contains mappings for keys appearing in context_features, and the second dictionary contains mappings for keys appearing in sequence_features.

需要至少提供一个非空的 ```context_features``` 或 ```sequence_features```

> At least one of context_features and sequence_features must be provided and non-empty.



。。。。。。。。。。。。。。。。。。未完待续。。。。。。。。。。。。。。。。。



## BATCHING AND PADDING DATA

Tensorflow的RNN函数需要一个tensor作为输入，形如[B, T, ...]，其中B是batch size，T是每个输入对应的的时间长度。
一般的，一个batch中的时间序列并不都是等长的，但是RNN要求等长，所以通过padding的方法(补0)实现

> Tensorflow’s RNN functions expect a tensor of shape [B, T, ...] as input, where B is the batch size and T is the length in time of each input (e.g. the number of words in a sentence). The last dimensions depend on your data. Do you see a problem with this? Typically, not all sequences in a single batch are of the same length T, but in order to feed them into the RNN they must be. Usually that’s done by padding them: Appending 0‘s to examples to make them equal in length.

batch padding解决了outlier的问题，因为每个等长处理是在同一个patch中进行的，因此outlier的影响程度只局限于一个batch内。

> Now imagine that one of your sequences is of length 1000, but the average length is 20. If you pad all your examples to length 1000 that would be a huge waste of space (and computation time)! That’s where batch padding comes in. If you create batches of size 32, you only need to pad examples within the batch to the same length (the maximum length of examples in that batch). That way, a really long example will only affect a single batch, not all of your data.




只要在调用 ```tf.train.batch``` 时，加入 ```dynamic_pad=True```, TensorFlow会自动完成这些batch padding的设置


> That all sounds pretty messy to deal with. Luckily, Tensorflow has built-in support for batch padding. If you set dynamic_pad=True when calling tf.train.batch the returned batch will be automatically padded with 0s. Handy! A lower-level option is to use tf.PaddingFIFOQueue.


SIDE NOTE: BE CAREFUL WITH 0’S IN YOUR VOCABULARY/CLASSES

> If you have a classification problem and your input tensors contain class IDs (0, 1, 2, …) then you need to be careful with padding. Because you are padding tensos with 0’s you may not be able to tell the difference between 0-padding and “class 0”. Whether or not this can be a problem depends on what your model does, but if you want to be safe it’s a good idea to not use “class 0” and instead start with “class 1”. (An example of where this may become a problem is in masking the loss function. More on that later).

















































实际使用中，经常把数据预处理为 ```tf.SequenceExample``` 并保存为TFRecord，再通过 ```tf.TFRecordReader``` 和 ```tf.parse_single_sequence_example``` 等读取和解析数据。

### 预处理成 ```tf.SequenceExample```

```python
import tensorflow as tf
import tempfile

sequences = [[1, 2, 3], [4, 5, 1], [1, 2]]
label_sequences = [[0, 1, 0], [1, 0, 0], [1, 1]]

# 把数据预处理为 tf.SequenceExample 对象
def make_example(sequence, labels):
    # The SequenceExample object we return
    ex = tf.train.SequenceExample()   # 最终返回一个SequenceExample对象
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    ex.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    fl_tokens = ex.feature_lists.feature_list["tokens"]
    fl_labels = ex.feature_lists.feature_list["labels"]
    for token, label in zip(sequence, labels):
        fl_tokens.feature.add().int64_list.value.append(token)
        fl_labels.feature.add().int64_list.value.append(label)
    return ex
```




### python tempfile

如何你的应用程序需要一个临时文件来存储数据，但不需要同其他程序共享，那么用TemporaryFile函数创建临时文件是最好的选择。其他的应用程序是无法找到或打开这个文件的，因为它并没有引用文件系统表。用这个函数创建的临时文件，关闭后会自动删除。


```python
    temp = tempfile.NamedTemporaryFile()
    temp.close()

    with tempfile.NamedTemporaryFile() as temp:
        ...

```

如果临时文件会被多个进程或主机使用，那么建立一个有名字的文件是最简单的方法。这就是NamedTemporaryFile要做的，可以使用name属性访问它的名字，尽管文件带有名字，但它仍然会在close后自动删除

### 使用 TFRecords 在TensorFlow 中实现对数据的高速存储
适用于数据量比较大的情况，无法将全部数据读入内存。
参考[这里](http://ycszen.github.io/2016/08/17/TensorFlow%E9%AB%98%E6%95%88%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE/)

一般步骤

*   TFRecords文件包含了tf.train.Example 协议内存(缓存)块(protocol buffer)(协议内存块包含了字段 Features)。我们可以写一段代码获取你的数据， 将数据填入到Example协议内存(缓存)块(protocol buffer)，将协议内存块序列化为一个字符串， 并且通过tf.python_io.TFRecordWriter 写入到TFRecords文件。

*   从TFRecords文件中读取数据， 可以使用tf.TFRecordReader的tf.parse_single_example解析器。这个操作可以将Example协议内存块(protocol buffer)解析为张量。

TFRecords其实是一种二进制文件，虽然它不如其他格式好理解，但是它能更好的利用内存，更方便复制和移动，并且不需要单独的标签文件（等会儿就知道为什么了）… …总而言之，这样的文件格式好处多多，所以让我们用起来吧。