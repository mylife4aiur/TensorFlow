# TensorFlow r1.0

## Neural Network ( tf.nn )

### Activation Functions 激活函数

*   Smooth nonlinearities 光滑非线性
    ```python
    tf.nn.sigmoid, tf.nn.tanh, tf.nn.elu, tf.nn.softplus, tf.nn.softsign
    ```
*   Continuous but not everywhere differentiable functon 连续不完全可导函数
    ```python
    tf.nn.relu, tf.nn.relu6, tf.nn.crelu, tf.nn.relu_x
    ```
*   Random regularization (Dropout)
    ```python
    tf.nn.dropout
    ```
![](F:\TensorFlow\activation_funcs1.png)

### Normalization

Normalization is useful to prevent neurons from saturating when inputs may have varying scale, and to aid generalization.

```python
tf.nn.l2_normalize
tf.nn.local_response_normalization
tf.nn.sufficient_statistics
tf.nn.normalize_moments
tf.nn.moments
tf.nn.weighted_moments
tf.nn.fused_batch_norm
tf.nn.batch_normalization
tf.nn.batch_norm_with_global_normalization
```

### Recurrent Neural Networks

```
tf.nn.dynamic_rnn
tf.nn.bidirectional_dynamic_rnn
tf.nn.raw_rnn
```

