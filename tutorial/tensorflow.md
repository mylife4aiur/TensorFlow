# Tensorflow
## Tutorials
### Vector Representations of Words
这个教程是基于 Mikolov 的[论文](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf).
这种模型被用于学习单词的向量表示，被称为“词嵌入”（word embedding）
#### 动机：为什么要学习词嵌入
在图像和音频处理系统中，会使用到信息量丰富的高维度数据，对于图片数据，它们被编码成用个体像素强度（individual raw pixel-intensities）组成的向量；对于音频数据，它们被表示为功率谱密度系数（power spectral density coefficients）的向量。
对于语音识别来说，完成这一任务的所有需要的信息都蕴含在数据中（因为人类能够基于这些数据完成这些任务）。但是，传统上，自然语言处理系统会把单词看作离散的原子符号，比如单词 ‘cat’ 会被表示为 ```Id537```，单词 ‘dog’ 会被表示为 ```Id143```。这种编码方式是十分随意的，无法为系统提供这些个体符号之间可能存在的关系信息。也就是说系统在处理关于 ‘dog’ 数据的时候，几乎无法利用已经习得的关于 ’cat’ 的知识。
使用唯一的，离散的 id 表示单词，进一步会导致数据变稀疏。这种情况通常意味着我们可能需要更多的数据来训练统计模型。
使用向量表示法可以一定程度上解决这一问题。

[向量空间模型](https://en.wikipedia.org/wiki/Vector_space_model)（VSM）将单词嵌入到一个连续的向量空间，在这个空间里，语义上相似的单词会被映射成相近的数据点。向量空间模型在 NLP 领域具有很长的历史，但是所有的这一类方法都以某种方式依赖于分布假设（ [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis)），这一假设认为出现在同一段文字中的单词会共享语义信息。利用这个原则的方法可以分类两类：基于计数的方法 （count-based methods）和预测模型（predictive methods）。

二类方法的具体区别可以参看 [Baroni et al.](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf) 的介绍，简单来说，基于计数的方法会在一个大的文本全集（text corpus）里面统计单词和其相邻单词一起出现的频率，然后将这些计数统计信息映射到一个小而密的向量。预测模型直接尝试根据一个单词的邻居来预测一个小而密的嵌入向量。

Word2vec 是一个专门为基于原始文本学习词嵌入向量而设计的高效率计算的预测模型。它包含两种模型，the Continuous Bag-of-Words model (CBOW) 和 the Skip-Gram model。这两种模型在算法层面上是相似的，差别仅在于 CBOW 基于原始文本中的单词来预测目标单词，而 skip-gram 相反，基于目标单词来预测原始文本中的单词。这种相反的策略看起来像随机选择，但是统计发现，CBOW 对于分布性的信息会做过度平滑的处理（因为它将整个上下文当作一个观测值）。
大多数情况下来说，CBOW 对于小数据集更有效。而 skip-gram 将每一个 context-target 对都看作是一个新的观测值，这会在大数据集上面表现得更好。在下面的教程中，我们会更关注 skip-gram 模型。

#### Scaling up with Noise-Contrastive Training
我们一般通过极大似然估计来训练神经概率语言模型（Neural probabilistic language model），以 softmax 函数为基础，最大化给出前文单词 $h$ 的情况下，下一个单词 $w_t$ 出现的概率。

$$P(w_t|h)=softmax(score(w_t,h))=\frac{e^{score(w_t,h)}}{\sum_{Word\ w'\ in\ Vocab}e^{score(w',h)}}$$

其中，$score(w_t,h)$ 计算单词 $w_t$ 和前文 $h$ 之间的相容性（通常使用点乘）。我们通过最大化训练集上的对数似然函数来训练这个模型。比如，最大化：
$$J_{ML}=\log P(w_t|h)=score(w_t,h)-\log (\sum_{Word\ w'\ in\ Vocab} e^{score(w',h)})$$

这种方法能为语言建模生成一个正确的归一化概率模型。但是这种方法的计算开销十分巨大，因为我们需要在每一步训练过程中，都要用到当前文本 $h$ 中的所有其他单词 $w'$的得分,来计算和归一化这个概率。

然而，在未来对 word2vec 的学习中，我们不需要一个使用完整数据的概率模型。CBOW 和 skip-gram 模型使用一个二分类目标函数（逻辑回归），基于相同的上下文信息，从k个虚构的单词（噪声）$\tilde{w_t}$ 中分辨出真实的目标单词 $w_t$。

数学表达上，这个目标函数是要最大化

$$J_{NEG}=\log Q_\theta(D=1|w_t,h)+k\mathbb{E}_{\tilde{w}\sim P_{noise}}[\log Q_\theta(D=0|\tilde{w},h)] $$

其中，$Q_\theta(D=1|w_t,h)$ 是指，单词 $w$ 在前文是 $h$ 的情况下，使用学习到的嵌入向量 $\theta$ 计算，属于数据集 $D$ 的二元逻辑回归概率。对于公式中的期望，在实际计算中，我们从噪声分布中抽取 k 个单词来估算（比如计算一个[蒙特卡洛均值](https://en.wikipedia.org/wiki/Monte_Carlo_integration)）

目标函数取到最大值的时候，模型分配给真实单词高概率，给噪声单词小概率。技术上，这被称为[负采样](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)。使用这个损失有数学上的好处：这种方式的更新方法近似了 softmax 函数更新的极限。这是十分有吸引力的，因为计算只涉及到我们所选择的噪声单词，而不是词汇表中的所有单词。因此这种方法的训练速度更快。我们在 ```tf.nn.nce_loss``` 函数中使用了一个非常相似的 noise-contrastive estimation (NCE) 损失函数。

#### Skip-gram 模型

举一个例子，我们考虑下面这个数据集：```the quick brown fox jumped over the lazy dog``` 。

首先构建一个数据集，包含单词和它所出现的上下文。我们可以定义任何有意义的“上下文”（context），事实上人们大多数人会选择基于句法的上下文（syntactic context）。比如目标单词左侧的单词，目标单词右侧的单词等等。我们这里选择最简单的定义，定义“上下文”为在目标单词左右窗口宽度内的所有单词。如果将窗口宽度设为1，那么我们得到了形如```(context, target)```的数据集：```([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...```

之前提到 skip-gram 会颠倒 contexts 和 target，并尝试基于目标单词预测每一个上下文的单词。也就是说任务变成了根据 'quick' 预测 'the' 和 'brown'。所以我们的数据集可以转换为 ```(input, output)``` 的形式：```(quick, the), (quick, brown), (brown, quick), (brown, fox), ...```。目标函数定义在整个数据集上，我们一般使用随机梯度下降（SGD）的方法，每次只使用一个样本做优化（或使用‘minibatch’）。

假设在训练第 $t$ 步的时候，我们观测到了上面这样的训练样本， 目标是通过 ```quick``` 来预测 ```the```。通过在某种噪声分布下采样，选择```num_noise``` 个噪声样本，通常使用 unigram distribution $P(w)$。简单起见我们令 ```num_noise=1```，并且使用 ```sheep``` 作为噪声样本。接下来我们计算这对观测样本和噪声样本的损失值，那么第 $t$ 步的目标函数变成了

$$J_{NEG}^{(t)}
=\log Q_\theta(D=1|the,quick)+\log Q_\theta(D=0|sheep,quick) $$

目标是对嵌入参数 $\theta$ 做更新，来提升目标函数（因为本例中需要最大化目标函数）。为此我们需要计算损失函数对嵌入参数 $\theta$ 的导数 $\frac{\partial}{\partial\theta}J_{NEG}$。
当我们在整个的数据集上面重复这个过程的时候，每个单词的嵌入向量会不停地被移动，直到能成功的从噪声单词中辨别出真实单词。

使用像[t-SNE](http://lvdmaaten.github.io/tsne/)一类的降维技术，我们可以把学习的到的向量映射到2维进行可视化。这些向量能捕捉到一些一般性的，有用的语义信息，以及各个单词之间的关系。有意思的是，在诱导向量空间（induced vector space）中，特定的方向对应着特定的语义关系。这也解释了了为什么这些向量也被用于很多经典 NLP 预测任务的特征。

#### 建立图（代码）
既然是关于嵌入的，先定义我们的嵌入矩阵，一开始它会是一个随机矩阵，我们用均匀分布来初始化嵌入矩阵。

```python
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
```
我们使用逻辑回归的形式定义噪音对比估计损失（noise-contrastive estimation loss）。为此，我们需要为单词表中的每一个单词定义权重（weight）和偏置项（bias）：

```python
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```
下面我们开始定义我们的 skip-gram 模型图。简单起见，假设我们将所有的文本全集都用一个整数化的词汇表表示，每一个单词对应成一个整数。skip-gram 模型接受两个输入。一个包含所有的上下文单词的整数表示，一个包含所有的目标单词。我们需要为这些输入建立 placeholder 节点，以便后续的数据传入。


```python
# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1]
```
我们需要查看输入的 batch 中每一个单词对应的向量，Tensorflow 中有一个函数可以简单地实现这一功能。

```python
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```

获得了每个单词的嵌入向量后，我们希望使用噪音对比估计损失来预测目标单词。

```python
# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size))
```

至此，我们得到了一个损失节点，下面需要计算梯度并更新参数。我们使用随机梯度下降法来计算梯度。

```python
# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
```

#### 训练模型
训练模型的操作十分简单，通过一个 ```feed_dict``` 将数据放进所有的 placeholder，通过在循环中调用 ```tf.Session.run``` 来训练模型。

#### 嵌入的可视化
我们使用 t-SNE 可视化学习到的嵌入向量。
#### 评价嵌入向量：Analogical Reasoning
一个评价嵌入向量好坏的简单方法是，直接使用它们预测句法和语义关系，如 ```king is to queen as father is to ?```。这被称作 analogical reasoning （类比推理），[Mikolov and colleagues](http://www.anthology.aclweb.org/N/N13/N13-1090.pdf) 介绍了相关的工作。

想要进一步研究这种评估的实现，参考[models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py)中的 ```build_eval_graph()``` 和 ```eval()``` 函数.

超参数的选择会极大地影响这一任务的准确性。为在这个任务上达到高水平的表现（state-of-the-art performance），需要在大数据集上多次训练，调参，还要使用一下小技巧（tricks），如对数据的下采样（subsample）。这里不再介绍。

#### 代码实现的优化
我们对训练嵌入向量的基本实现体现了 TensorFlow 的灵活性。比如只要把对 ```tf.nn.nce_loss()``` d的调用换成其他的现成选择，如 ```tf.nn.sampled_softmax_loss()```，就可以实现改变目标函数。你也可以在 TensorFlow 中自己手写一个新的目标函数，然后让优化器计算它的导数。

当你的数据过大，模型的计算速度受限于输入数据的读入阶段时，可以考虑直接使用 TensorFlow 提供的数据读取模块，因为相比于 Python，TensorFlow 做数据读取后端工作量会更小，这部分在 [New Data Formats](https://www.tensorflow.org/extend/new_data_formats) 中会介绍。

此外还可以便携自己的优化函数（TensorFlow Ops），就像 [Adding a New Op](https://www.tensorflow.org/extend/adding_an_op) 中所描述的



###
###
## function
### ShareVariable
### Slice
### scan

```python
tf.scan(
    fn,
    elems,
    initializer = None,
    parallel_iterations = 10,
    back_prop = True,
    swap_memory = False,
    infer_shape = True,
    name = None
)
```
scan 函数作用于从 ```elems``` 参数中所拆包（unpack）得到的张量（tensor）列表的第0个维度。

最简单版本的 ```scan``` 函数会重复地把可调用函数 ```fn``` 从头到尾地应用于一个序列。这个序列是由从 ```elems``` 得到的张量中，第0个维度的所有元素组成的。

可调用函数 ```fn``` 接受两个张量作为参数，第一个参数是由调用前一个 ```fn``` 得到的累积值。如果 ```initializer``` 是空值的话（None），那么 ```elems``` 必须包含至少一个元素，其中的第一个元素会被用来当作初始值。

假设 ```elems``` 被拆包成 ```values```，一个张量的列表。那么返回结果的形状会是 ```[len(values)] + fn(initializer, values[0]).shape```

> 这里其实最简单的情况是，假设 ```elems``` 只由一个张量 A 组成，那么拆包得到的 ```values = [A]```

----------------这里可能有点问题----------------

这一方法允许以元组或列表的方式传入多个 ```elems``` 参数和累加器（这里是指```fn```的第一个输出值）。如果使用这种传参方法，中的所有张量的第一个维度都要相同。```fn``` 的第二个参数要和 ```elems``` 的结构相匹配

如果没有提供 ```initializer``` 参数，```fn``` 的输出结构和类型应该和它的输入一致；在这种情况下，```fn``` 的第二个参数要和 ```elems``` 的结构一致。如果提供了 ```initializer``` 参数，那么 ```fn``` 的第一个参数和输出都应该和其保持一致。

比如，如果 ```elems``` 是 ```(t1,[t2,t3])``` 而 ```initializer``` 是 ```[i1,i2]``` 那么一个可能的 ```fn``` 函数是 ```fn = lambda (acc_p1, acc_p2), (t1 [t2, t3]):```，这一函数的返回值也必须是列表 ```[acc_p1, acc_p2]```。


https://rdipietro.github.io/tensorflow-scan-examples/

```python
def fn(previous_output, current_input):
    return previous_output + current_input
elems = tf.Variable([1.0, 2.0, 2.0, 2.0])
elems = tf.identity(elems)
initializer = tf.constant(0.0)
out = tf.scan(fn, elems, initializer=initializer)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(out))
```





### map_fn
对于从 ```elems``` 拆包得到的张量列表的第0维上面的 map 操作
