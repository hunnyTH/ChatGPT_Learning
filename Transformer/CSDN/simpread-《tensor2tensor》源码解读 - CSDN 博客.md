> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/T940842933/article/details/140524645?spm=1001.2014.3001.5502)

**目录**

[说明](#%E4%B8%80%E3%80%81%E8%AF%B4%E6%98%8E)

[Transformer 模型框架](#t0)

[一、自注意力机制（Self-attention）](#t1)

[1. 缩放点积注意力机制（Scaled dot-product attention）](#t2)

[2. 多头注意力机制（Multi-head attention）](#t3)

[3. 注意力计算](#t4)

[二、编码前准备](#t5)

[1. Inputs & Outputs（Shift right）](#t6) 

[1.1 data_reader](#t7)

[2. Shift right](#t8)

[2. Embedding](#t9)

[3. Position Encoding](#t10)

[4. transformer_prepare_encoder](#t11)

[三、编码器（Encoder）](#t12)

[四、解码前准备](#t13)

[五、解码器（Decoder）](#t14)

[1. 单个解码器层](#t15)

[2. 解码器](#t16)

[六、Softmax](#t17)

[1. Saturating Sigmoid](#t18)

[2. Hard Sigmoid](#t19)

[七、transformer](#t20)

**说明**

本篇博客记录学习项目《tensor2tensor》的全过程，不做任何商业用途，如有侵权请及时联系。

*   论文链接：[https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf "https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf")
*   项目地址：[https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor "https://github.com/tensorflow/tensor2tensor")
*   学习工具：知云文献翻译 V8.4、Visual Studio Code、ChatGPT-4o-mini
*   学习内容：代码解读
*   学习目标：分析 transformer 框架源码，理解模型算法过程

Transformer 模型框架
----------------

![](https://i-blog.csdnimg.cn/direct/4ef477fd464f40abb2918c1d4e9c0a03.png)

一、自注意力机制（Self-attention）
------------------------

### 1. 缩放点积注意力机制（Scaled dot-product attention）

**路径：**tensor2tensor\layers\common_attention.py

```
def scaled_dot_product_attention_simple(q, k, v, bias, name=None):
  """Scaled dot-product attention. One head. One spatial dimension.
  Args:
    q: a Tensor with shape [batch, length_q, depth_k]
    k: a Tensor with shape [batch, length_kv, depth_k]
    v: a Tensor with shape [batch, length_kv, depth_v]
    bias: optional Tensor broadcastable to [batch, length_q, length_kv]
    name: an optional string
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_):
    scalar = tf.rsqrt(tf.to_float(common_layers.shape_list(q)[2]))
    logits = tf.matmul(q * scalar, k, transpose_b=True)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, )
    if common_layers.should_generate_summaries():
      tf.summary.image(
          "attention", tf.expand_dims(tf.pow(weights, 0.2), 3), max_outputs=1)
    return tf.matmul(weights, v)
```

**函数功能**

`scaled_dot_product_attention_simple` 函数实现了单头的缩放点积注意力机制，适用于具有一个空间维度的输入。该机制通过计算查询（Q）和键（K）之间的相似度来生成注意力权重，并使用这些权重对值（V）进行加权求和。

**参数说明**

*   `q`: 查询张量，形状为 `[batch, length_q, depth_k]`，其中 `batch` 是批量大小，`length_q` 是查询的长度，`depth_k` 是查询的深度。
*   `k`: 键张量，形状为 `[batch, length_kv, depth_k]`，`length_kv` 是键值对的长度。
*   `v`: 值张量，形状为 `[batch, length_kv, depth_v]`，`depth_v` 是值的深度。
*   `bias`: 可选的偏置张量，形状应可广播到 `[batch, length_q, length_kv]`。
*   `name`: 可选的字符串，指定变量作用域的名称。

**返回值**

*   返回一个张量，表示经过注意力机制处理后的值，形状为 `[batch, length_q, depth_v]`。

**逻辑**

1.  使用 `tf.variable_scope` 来定义作用域，便于管理变量。计算缩放因子 `scalar`，它是查询深度的平方根的倒数。这个缩放因子用于防止在计算点积时，随着深度增加而导致的梯度消失问题。
    
    ```
    with tf.variable_scope(name, default_):
        scalar = tf.rsqrt(tf.to_float(common_layers.shape_list(q)[2]))
    ```
    
2.  计算 `logits`，即查询和键的点积，使用 `tf.matmul` 进行矩阵乘法。由于 `q` 被缩放了，因此可以有效地控制注意力的强度。
    
    ```
        logits = tf.matmul(q * scalar, k, transpose_b=True)
    
    
    ```
    
3.  如果提供了偏置，则将其添加到 `logits` 中，以便调整注意力分布。
    
    ```
        if bias is not None:
            logits += bias
    ```
    
4.  计算注意力权重，使用 `softmax` 函数将 `logits` 转换为概率分布。
    
    ```
        weights = tf.nn.softmax(logits, )
    
    
    ```
    
5.  如果需要生成摘要（用于可视化），则将注意力权重进行处理并记录为图像。
    
    ```
        if common_layers.should_generate_summaries():
            tf.summary.image(
                "attention", tf.expand_dims(tf.pow(weights, 0.2), 3), max_outputs=1)
    ```
    
6.  最后，使用计算得到的权重对值 `v` 进行加权求和，返回最终的输出张量。
    
    ```
        return tf.matmul(weights, v)
    
    
    ```
    

### 2. 多头注意力机制（Multi-head attention）

**路径：**tensor2tensor\layers\common_attention.py

```
def multihead_self_attention_memory_efficient(x,
                                              bias,
                                              num_heads,
                                              head_size=None,
                                              epsilon=1e-6,
                                              forget=True,
                                              test_vars=None,
                                              name=None):
  """Multihead scaled-dot-product self-attention.
  Includes layer norm.
  Returns multihead-self-attention(layer_norm(x))
  Computes one attention head at a time to avoid exhausting memory.
  If forget=True, then forget all forwards activations and recompute on
  the backwards pass.
  Args:
    x: a Tensor with shape [batch, length, input_size]
    bias: an attention bias tensor broadcastable to [batch, 1, length, length]
    num_heads: an integer
    head_size: an optional integer - defaults to input_size/num_heads
    epsilon: a float, for layer norm
    forget: a boolean - forget forwards activations and recompute on backprop
    test_vars: optional tuple of variables for testing purposes
    name: an optional string
  Returns:
    A Tensor.
  """
  io_size = x.get_shape().as_list()[-1]
  if head_size is None:
    assert io_size % num_heads == 0
    head_size = io_size / num_heads
 
  def forward_internal(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
    """Forward function."""
    n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
    wqkv_split = tf.unstack(wqkv, num=num_heads)
    wo_split = tf.unstack(wo, num=num_heads)
    y = 0
    for h in range(num_heads):
      with tf.control_dependencies([y] if h > 0 else []):
        combined = tf.nn.conv1d(n, wqkv_split[h], 1, "SAME")
        q, k, v = tf.split(combined, 3, axis=2)
        o = scaled_dot_product_attention_simple(q, k, v, attention_bias)
        y += tf.nn.conv1d(o, wo_split[h], 1, "SAME")
    return y
 
  key = (
      "multihead_self_attention_memory_efficient %s %s" % (num_heads, epsilon))
  if not forget:
    forward_fn = forward_internal
  elif key in _function_cache:
    forward_fn = _function_cache[key]
  else:
 
    @function.Defun(compiled=True)
    def grad_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias, dy):
      """Custom gradient function."""
      with tf.control_dependencies([dy]):
        n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
        wqkv_split = tf.unstack(wqkv, num=num_heads)
        wo_split = tf.unstack(wo, num=num_heads)
        deps = []
        dwqkvs = []
        dwos = []
        dn = 0
        for h in range(num_heads):
          with tf.control_dependencies(deps):
            combined = tf.nn.conv1d(n, wqkv_split[h], 1, "SAME")
            q, k, v = tf.split(combined, 3, axis=2)
            o = scaled_dot_product_attention_simple(q, k, v, attention_bias)
            partial_y = tf.nn.conv1d(o, wo_split[h], 1, "SAME")
            pdn, dwqkvh, dwoh = tf.gradients(
                ys=[partial_y],
                xs=[n, wqkv_split[h], wo_split[h]],
                grad_ys=[dy])
            dn += pdn
            dwqkvs.append(dwqkvh)
            dwos.append(dwoh)
            deps = [dn, dwqkvh, dwoh]
        dwqkv = tf.stack(dwqkvs)
        dwo = tf.stack(dwos)
        with tf.control_dependencies(deps):
          dx, dnorm_scale, dnorm_bias = tf.gradients(
              ys=[n], xs=[x, norm_scale, norm_bias], grad_ys=[dn])
        return (dx, dwqkv, dwo, tf.zeros_like(attention_bias), dnorm_scale,
                dnorm_bias)
 
    @function.Defun(
        grad_func=grad_fn, compiled=True, separate_compiled_gradients=True)
    def forward_fn(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
      return forward_internal(x, wqkv, wo, attention_bias, norm_scale,
                              norm_bias)
 
    _function_cache[key] = forward_fn
 
  if bias is not None:
    bias = tf.squeeze(bias, 1)
  with tf.variable_scope(name, default_, values=[x]):
    # TODO(noam): it would be nice to save memory by casting x to float16
    # here, but this causes problems with the gradients.  Figure out if there
    # is a way to leave the gradients as float32.
    if test_vars is not None:
      wqkv, wo, norm_scale, norm_bias = list(test_vars)
    else:
      wqkv = tf.get_variable(
          "wqkv", [num_heads, 1, io_size, 3 * head_size],
          initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
      wo = tf.get_variable(
          "wo", [num_heads, 1, head_size, io_size],
          initializer=tf.random_normal_initializer(
              stddev=(head_size * num_heads)**-0.5))
      norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
    y = forward_fn(x, wqkv, wo, bias, norm_scale, norm_bias)
    y.set_shape(x.get_shape())
    return y
```

**函数功能**

multihead_self_attention_memory_efficient 函数实现了多头自注意力机制，采用了内存高效的计算方式，逐个头进行计算以避免内存耗尽。它结合了层归一化，并提供了可选的前向激活遗忘机制，以便在反向传播时节省内存。这个实现适用于大规模模型的训练，尤其是在资源有限的情况下。

**参数说明**

*   `x`: 输入张量，形状为 `[batch, length, input_size]`，其中 `batch` 是批量大小，`length` 是序列长度，`input_size` 是输入特征的维度。
*   `bias`: 可选的注意力偏置张量，形状应可广播到 `[batch, 1, length, length]`。
*   `num_heads`: 整数，表示注意力头的数量。
*   `head_size`: 可选整数，表示每个头的大小，默认为 `input_size / num_heads`。
*   `epsilon`: 用于层归一化的浮点数。
*   `forget`: 布尔值，指示是否在反向传播时忘记所有前向激活并重新计算。
*   `test_vars`: 可选的变量元组，用于测试目的。
*   `name`: 可选的字符串，指定变量作用域的名称。

**返回值**

*   返回一个张量，表示经过多头自注意力机制处理后的输出，形状与输入 `x` 相同。

**逻辑**

1.  **计算头大小**：
    
    ```
    io_size = x.get_shape().as_list()[-1]
    if head_size is None:
        assert io_size % num_heads == 0
        head_size = io_size / num_heads
    ```
    
    *   获取输入张量的最后一维大小 `io_size`。
    *   如果未指定 `head_size`，则计算每个头的大小。
2.  **定义前向计算函数**：
    
    ```
    def forward_internal(x, wqkv, wo, attention_bias, norm_scale, norm_bias):
        """Forward function."""
        n = common_layers.layer_norm_compute(x, epsilon, norm_scale, norm_bias)
        wqkv_split = tf.unstack(wqkv, num=num_heads)
        wo_split = tf.unstack(wo, num=num_heads)
        y = 0
        for h in range(num_heads):
            with tf.control_dependencies([y] if h > 0 else []):
                combined = tf.nn.conv1d(n, wqkv_split[h], 1, "SAME")
                q, k, v = tf.split(combined, 3, axis=2)
                o = scaled_dot_product_attention_simple(q, k, v, attention_bias)
                y += tf.nn.conv1d(o, wo_split[h], 1, "SAME")
        return y
    ```
    
    *   对输入 `x` 进行层归一化。
    *   将权重张量 `wqkv` 和 `wo` 拆分为多个头。
    *   对每个头进行卷积操作，计算查询、键、值，并调用缩放点积注意力函数。
    *   将所有头的输出相加，得到最终输出。
3.  **缓存和梯度计算**：
    
    ```
    key = ("multihead_self_attention_memory_efficient %s %s" % (num_heads, epsilon))
    if not forget:
        forward_fn = forward_internal
    elif key in _function_cache:
        forward_fn = _function_cache[key]
    else:
        ...
        _function_cache[key] = forward_fn
    ```
    
    *   根据 `forget` 标志决定是否缓存前向计算函数。
    *   如果 `forget` 为 `True`，则定义自定义梯度函数 `grad_fn`，以便在反向传播时重新计算前向激活。
4.  **权重初始化**：
    
    ```
    if bias is not None:
        bias = tf.squeeze(bias, 1)
    with tf.variable_scope(name, default_, values=[x]):
        if test_vars is not None:
            wqkv, wo, norm_scale, norm_bias = list(test_vars)
        else:
            wqkv = tf.get_variable(
                "wqkv", [num_heads, 1, io_size, 3 * head_size],
                initializer=tf.random_normal_initializer(stddev=io_size**-0.5))
            wo = tf.get_variable(
                "wo", [num_heads, 1, head_size, io_size],
                initializer=tf.random_normal_initializer(
                    stddev=(head_size * num_heads)**-0.5))
            norm_scale, norm_bias = common_layers.layer_norm_vars(io_size)
        y = forward_fn(x, wqkv, wo, bias, norm_scale, norm_bias)
        y.set_shape(x.get_shape())
        return y
    ```
    
    *   初始化权重和层归一化参数。
    *   调用前向函数计算输出 `y`，并返回。

### 3. 注意力计算

路径：tensor2tensor\layers\common_attention.py

```
def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product",
                        max_relative_position=None,
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        image_shapes=None,
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        ,
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,
                        layer_collection=None,
                        recurrent_memory=None,
                        chunk_number=None,
                        hard_attention_k=0,
                        gumbel_noise_weight=0.0,
                        max_area_width=1,
                        max_area_height=1,
                        memory_height=1,
                        area_key_mode="mean",
                        area_value_mode="sum",
                        training=True,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    heads_share_relative_embedding: boolean to share relative embeddings
    add_relative_to_values: a boolean for whether to add relative component to
                            values.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory: An optional transformer_memory.RecurrentMemory, which
      retains state across chunks. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
    gumbel_noise_weight: if > 0, apply Gumbel noise with weight
      `gumbel_noise_weight` before picking top-k. This is a no op if
      hard_attention_k <= 0.
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    training: indicating if it is in the training mode.
    **kwargs (dict): Parameters for the attention function.
  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.
    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.
  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  vars_3d_num_heads = num_heads if vars_3d else 0
 
  if layer_collection is not None:
    if cache is not None:
      raise ValueError("KFAC implementation only supports cache is None.")
    if vars_3d:
      raise ValueError("KFAC implementation does not support 3d vars.")
 
  if recurrent_memory is not None:
    if memory_antecedent is not None:
      raise ValueError("Recurrent memory requires memory_antecedent is None.")
    if cache is not None:
      raise ValueError("Cache is not supported when using recurrent memory.")
    if vars_3d:
      raise ValueError("3d vars are not supported when using recurrent memory.")
    if layer_collection is not None:
      raise ValueError("KFAC is not supported when using recurrent memory.")
    if chunk_number is None:
      raise ValueError("chunk_number is required when using recurrent memory.")
 
  with tf.variable_scope(name, default_,
                         values=[query_antecedent, memory_antecedent]):
 
    if recurrent_memory is not None:
      (
          recurrent_memory_transaction,
          query_antecedent, memory_antecedent, bias,
      ) = recurrent_memory.pre_attention(
          chunk_number,
          query_antecedent, memory_antecedent, bias,
      )
 
    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads,
                            layer_collection=layer_collection)
    if cache is not None:
      if attention_type not in ["dot_product", "dot_product_relative"]:
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")
 
      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        q_filter_width, q_padding, "q",
                                        vars_3d_num_heads=vars_3d_num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])
 
    q = split_heads(q, num_heads)
    if cache is None:
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)
 
    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5
 
    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      if max_area_width > 1 or max_area_height > 1:
        x = area_attention.dot_product_area_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            dropout_broadcast_dims=dropout_broadcast_dims,
            max_area_width=max_area_width,
            max_area_height=max_area_height,
            memory_height=memory_height,
            area_key_mode=area_key_mode,
            area_value_mode=area_value_mode,
            training=training)
      else:
        x = dot_product_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            make_image_summary=make_image_summary,
            dropout_broadcast_dims=dropout_broadcast_dims,
            activation_dtype=kwargs.get("activation_dtype"),
            hard_attention_k=hard_attention_k,
            gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_relative":
      x = dot_product_attention_relative(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          cache=cache is not None,
          allow_memory=recurrent_memory is not None,
          hard_attention_k=hard_attention_k,
          gumbel_noise_weight=gumbel_noise_weight)
    elif attention_type == "dot_product_unmasked_relative_v2":
      x = dot_product_unmasked_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "dot_product_relative_v2":
      x = dot_product_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "local_within_block_mask_right":
      x = masked_within_block_local_attention_1d(
          q, k, v, block_length=block_length)
    elif attention_type == "local_relative_mask_right":
      x = masked_relative_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary,
          dropout_rate=dropout_rate,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values,
          )
    elif attention_type == "local_mask_right":
      x = masked_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary)
    elif attention_type == "local_unmasked":
      x = local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                           gap_size, num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = dilated_self_attention_1d(q, k, v, block_length, block_width,
                                    gap_size, num_memory_blocks)
    x = combine_heads(x)
 
    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])
 
    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, ,
          layer_collection=layer_collection)
 
    if recurrent_memory is not None:
      x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x
```

**函数功能**

`multihead_attention` 函数实现了多头缩放点积注意力机制，并进行输入 / 输出的变换。

**参数说明**

*   **query_antecedent**: 查询张量，形状为 `[batch, length_q, channels]`。
*   **memory_antecedent**: 记忆张量，形状为 `[batch, length_m, channels]` 或 `None`。
*   **bias**: 注意力偏置张量。
*   **total_key_depth**: 整数，表示键的总深度。
*   **total_value_depth**: 整数，表示值的总深度。
*   **output_depth**: 整数，表示输出的深度。
*   **num_heads**: 整数，表示注意力头的数量，必须能够整除 `total_key_depth` 和 `total_value_depth`。
*   **dropout_rate**: 浮点数，表示丢弃率。
*   **attention_type**: 字符串，表示注意力类型，可以是多种类型，如 `"dot_product"`、`"dot_product_relative"` 等。
*   **max_relative_position**: 整数，表示生成唯一关系嵌入的最大距离，仅在使用相对位置注意力时相关。
*   **training**: 布尔值，指示是否处于训练模式。
*   **kwargs**: 其他参数，供注意力函数使用。

**返回值**

*   返回注意力变换的结果，输出形状为 `[batch_size, length_q, hidden_dim]`，如果提供了缓存字典，则输出形状为 `[batch_size, 1, hidden_dim]`。

**错误处理**

*   如果键的深度或值的深度不能被头的数量整除，将引发 `ValueError`。
*   如果使用了缓存且注意力类型不支持缓存，将引发 `NotImplementedError`。

**代码逻辑概述**

1.  **参数验证**：检查 `total_key_depth` 和 `total_value_depth` 是否能被 `num_heads` 整除。
2.  **变量定义**：根据输入参数定义变量和张量。
3.  **计算 Q、K、V**：根据输入的查询和记忆张量计算查询、键和值。
4.  **处理缓存**：如果提供了缓存，处理缓存中的键和值。
5.  **执行注意力计算**：根据指定的注意力类型计算注意力输出。
6.  **输出变换**：将注意力输出通过全连接层或其他方式进行变换。
7.  **返回结果**：返回最终的注意力输出。

二、编码前准备
-------

### 1. Inputs & Outputs（Shift right） 

#### 1.1 data_reader

**路径：**tensor2tensor\utils\data_reader.py

```
def input_fn(dataset,
             filepattern,
             skip_random_fraction_when_training,
             batch_size_means_tokens_param,
             batch_size_multiplier,
             max_length,
             mode,
             hparams,
             data_dir=None,
             params=None,
             config=None,
             force_repeat=False,
             prevent_repeat=False):
  """Builds input pipeline for problem.
  Args:
    dataset: the dataset to make input function from.
    filepattern: the pattern of files to read from.
    skip_random_fraction_when_training: whether to skip randomly when training.
    batch_size_means_tokens_param: whether batch size should mean tokens.
    batch_size_multiplier: how to multiply batch size when bucketing.
    max_length: maximum length,
    mode: tf.estimator.ModeKeys
    hparams: HParams, model hparams
    data_dir: str, data directory; if None, will use hparams.data_dir
    params: dict, may include "batch_size"
    config: RunConfig; should have the data_parallelism attribute if not using
      TPU
    force_repeat: bool, whether to repeat the data even if not training
    prevent_repeat: bool, whether to not repeat when in training mode.
      Overrides force_repeat.
  Returns:
    (features_dict<str name, Tensor feature>, Tensor targets)
  """
  is_training = mode == tf_estimator.ModeKeys.TRAIN
  if config and config.use_tpu:
    num_threads = 64
  else:
    num_threads = cpu_count() if is_training else 1
 
  if config and hasattr(config,
                        "data_parallelism") and config.data_parallelism:
    num_shards = config.data_parallelism.n
  else:
    num_shards = 1
 
  mlperf_log.transformer_print(
      key=mlperf_log.INPUT_MAX_LENGTH, value=max_length)
 
  def tpu_valid_size(example):
    return example_valid_size(example, hparams.min_length, max_length)
 
  def gpu_valid_size(example):
    drop_long_sequences = is_training or hparams.eval_drop_long_sequences
    max_validate_length = max_length if drop_long_sequences else 10**9
    return example_valid_size(example, hparams.min_length, max_validate_length)
 
  def define_shapes(example):
    batch_size = config and config.use_tpu and params["batch_size"]
    return standardize_shapes(example, batch_size=batch_size)
 
  # Read and preprocess
  data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)
 
  if (force_repeat or is_training) and not prevent_repeat:
    # Repeat and skip a random number of records
    dataset = dataset.repeat()
 
  if is_training and skip_random_fraction_when_training:
    data_files = contrib.slim().parallel_reader.get_data_files(filepattern)
    #  In continuous_train_and_eval when switching between train and
    #  eval, this input_fn method gets called multiple times and it
    #  would give you the exact same samples from the last call
    #  (because the Graph seed is set). So this skip gives you some
    #  shuffling.
    dataset = skip_random_fraction(dataset, data_files[0])
 
  dataset = dataset.map(cast_ints_to_int32, num_parallel_calls=num_threads)
 
  if batch_size_means_tokens_param:
    batch_size_means_tokens = True
  else:
    if _are_shapes_fully_defined(dataset.output_shapes):
      batch_size_means_tokens = False
    else:
      tf.logging.warning(
          "Shapes are not fully defined. Assuming batch_size means tokens.")
      batch_size_means_tokens = True
 
  # Batching
  if not batch_size_means_tokens:
    # Batch size means examples per datashard.
    if config and config.use_tpu:
      # on TPU, we use params["batch_size"], which specifies the number of
      # examples across all datashards
      batch_size = params["batch_size"]
      dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
      batch_size = hparams.batch_size * num_shards
      dataset = dataset.batch(batch_size)
  else:
    # batch_size means tokens per datashard
    if config and config.use_tpu:
      dataset = dataset.filter(tpu_valid_size)
      padded_shapes = pad_for_tpu(dataset.output_shapes, hparams, max_length)
      # on TPU, we use params["batch_size"], which specifies the number of
      # examples across all datashards
      batch_size = params["batch_size"]
      if hparams.pad_batch:
        tf.logging.warn(
            "Padding the batch to ensure that remainder eval batches are "
            "processed. This may lead to incorrect metrics for "
            "non-zero-padded features, e.g. images. Use a smaller batch "
            "size that has no remainder in that case.")
        dataset = dataset.padded_batch(
            batch_size, padded_shapes, drop_remainder=False)
        dataset = dataset.map(
            functools.partial(pad_batch, batch_multiple=batch_size),
            num_parallel_calls=num_threads)
      else:
        dataset = dataset.padded_batch(
            batch_size, padded_shapes, drop_remainder=True)
    else:
      # On GPU, bucket by length
      dataset = dataset.filter(gpu_valid_size)
      cur_batching_scheme = hparams_to_batching_scheme(
          hparams,
          shard_multiplier=num_shards,
          length_multiplier=batch_size_multiplier)
      if hparams.use_fixed_batch_size:
        # Here  batch_size really means examples per datashard.
        cur_batching_scheme["batch_sizes"] = [hparams.batch_size]
        cur_batching_scheme["boundaries"] = []
      dataset = dataset.apply(
          tf.data.experimental.bucket_by_sequence_length(
              example_length, cur_batching_scheme["boundaries"],
              cur_batching_scheme["batch_sizes"]))
 
      if not is_training:
        batch_multiple = num_shards
        if hparams.use_fixed_batch_size:
          # Make sure the last batch has the same fixed size as the rest.
          batch_multiple *= hparams.batch_size
        if batch_multiple > 1:
          tf.logging.warn(
              "Padding the batch to ensure that remainder eval batches have "
              "a batch size divisible by the number of data shards. This may "
              "lead to incorrect metrics for non-zero-padded features, e.g. "
              "images. Use a single datashard (i.e. 1 GPU) in that case.")
          dataset = dataset.map(
              functools.partial(pad_batch, batch_multiple=batch_multiple),
              num_parallel_calls=num_threads)
 
  dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)
 
  # Add shuffling for training batches. This is necessary along with record
  # level shuffling in the dataset generation. Record shuffling will shuffle
  # the examples. However, in some cases, it's possible that the shuffle
  # buffer size for record shuffling is smaller than the batch size. In such
  # cases, adding batch shuffling ensures that the data is in random order
  # during training
  if (is_training and hasattr(hparams, "batch_shuffle_size") and
      hparams.batch_shuffle_size):
    dataset = dataset.shuffle(hparams.batch_shuffle_size)
 
  # Split batches into chunks if targets are too long.
  # The new "chunk_number" feature is 0 for the first chunk and goes up then.
  # Chunks are reversed so the 0th chunk comes first, then the 1st and so on,
  # so models can attend to them in the order they arrive. The last chunk is
  # usually the one containing the end of the target sentence (EOS).
  chunk_length = hparams.get("split_targets_chunk_length", 0)
  max_chunks = hparams.get("split_targets_max_chunks", 100)
  if chunk_length > 0:
    def is_nonzero_chunk(example):
      """A chunk is zero if all targets are 0s."""
      return tf.less(0, tf.reduce_sum(tf.abs(example["targets"])))
 
    def split_on_length(example):
      """Split a batch of ditcs on length."""
      x = example["targets"]
      # TODO(kitaev): This code breaks if chunk_length * max_chunks < batch_size
      length_diff = chunk_length * max_chunks - tf.shape(x)[1]
      padded_x = tf.pad(x, [(0, 0), (0, length_diff), (0, 0), (0, 0)])
      chunks = [padded_x[:, i*chunk_length:(i+1)*chunk_length, :, :]
                for i in range(max_chunks - 1)]
      chunks.append(padded_x[:, (max_chunks - 1)*chunk_length:, :, :])
      new_example = {}
      # Setting chunk_number to be tf.range(max_chunks) is incompatible with TPU
      new_example["chunk_number"] = tf.concat([
          tf.expand_dims(tf.ones_like(c) * n, axis=0)
          for n, c in enumerate(chunks)
      ],
                                              axis=0)
      new_example["targets"] = tf.concat(
          [tf.expand_dims(c, axis=0) for c in chunks], axis=0)
      for k in example:
        if k != "targets":
          assert k != "chunk_number", (
              "Chunking code expects the chunk_number feature name to be "
              "available"
          )
          new_example[k] = tf.concat(
              [tf.expand_dims(example[k], axis=0) for _ in range(max_chunks)],
              axis=0)
      return tf.data.Dataset.from_tensor_slices(new_example)
 
    dataset = dataset.flat_map(split_on_length)
    dataset = dataset.filter(is_nonzero_chunk)
 
    # The chunking data pipeline thus far creates batches of examples where all
    # of the examples have the same chunk number. This can lead to periodic
    # fluctuations in the loss; for example, when all examples in the batch have
    # chunk number 0 the loss may be higher than midway through a sequence.
    # Enabling split_targets_strided_training adjusts the data so that each
    # batch includes examples at various points within a sequence.
    if is_training and hparams.split_targets_strided_training:
      # TODO(kitaev): make sure that shape inference works on GPU, not just TPU.
      inferred_batch_size = dataset.output_shapes["targets"].as_list()[0]
      if inferred_batch_size is None:
        raise ValueError(
            "Strided training is only implemented when the batch size can be "
            "inferred statically, for example when training on TPU."
        )
      chunk_stride = inferred_batch_size * max(
          1, max_chunks // inferred_batch_size) + 1
 
      def collapse_nested_datasets(example):
        """Converts a dataset of datasets to a dataset of tensor features."""
        new_example = {}
        for k, v in example.items():
          v = tf.data.experimental.get_single_element(
              v.batch(inferred_batch_size, drop_remainder=True))
          new_example[k] = v
        return tf.data.Dataset.from_tensor_slices(new_example)
 
      dataset = dataset.unbatch()
      dataset = dataset.window(inferred_batch_size, inferred_batch_size,
                               chunk_stride)
      dataset = dataset.flat_map(collapse_nested_datasets)
      dataset = dataset.batch(inferred_batch_size, drop_remainder=True)
 
  def prepare_for_output(example):
    if not config or not config.use_tpu:
      _summarize_features(example, num_shards)
    if mode == tf_estimator.ModeKeys.PREDICT:
      example["infer_targets"] = example.pop("targets")
      return example
    else:
      return example, example[hparams.get(
          key="labels_feature_name", default="targets")]
 
  dataset = dataset.map(prepare_for_output, num_parallel_calls=num_threads)
  dataset = dataset.prefetch(2)
 
  if mode == tf_estimator.ModeKeys.PREDICT:
    # This is because of a bug in the Estimator that short-circuits prediction
    # if it doesn't see a QueueRunner. DummyQueueRunner implements the
    # minimal expected interface but does nothing.
    tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, DummyQueueRunner())
 
  return dataset
```

**函数功能**

`input_fn` 函数构建了一个输入数据管道，处理数据集并为模型训练、评估或预测准备输入。它支持多种配置选项，以适应不同的训练模式（如 TPU、GPU）和数据处理需求。

**参数说明**

*   **dataset**: 输入的数据集对象，通常是一个 TensorFlow Dataset。
*   **filepattern**: 要读取的文件模式，通常用于指定数据文件的路径。
*   **skip_random_fraction_when_training**: 在训练时是否随机跳过部分数据。
*   **batch_size_means_tokens_param**: 批量大小是否表示为 tokens。
*   **batch_size_multiplier**: 在分桶时如何乘以批量大小。
*   **max_length**: 输入序列的最大长度。
*   **mode**: 模式，通常是 `tf.estimator.ModeKeys` 的一个值（如 TRAIN、EVAL、PREDICT）。
*   **hparams**: 模型超参数，包含训练所需的各种设置。
*   **data_dir**: 数据目录，如果为 None，则使用 `hparams.data_dir`。
*   **params**: 包含可能的参数字典，例如 "batch_size"。
*   **config**: 运行配置，包含数据并行性等属性。
*   **force_repeat**: 是否强制重复数据，即使不是训练模式。
*   **prevent_repeat**: 在训练模式下是否防止重复。

**返回值**

*   返回一个元组 `(features_dict, targets)`，其中 `features_dict` 是特征字典，`targets` 是目标张量。

**代码逻辑**

1.  **模式判断**：根据 `mode` 参数判断当前是训练、评估还是预测模式。
2.  **线程数设置**：根据是否使用 TPU 和训练模式设置并行线程数。
3.  **数据重复**：如果是训练模式且允许重复，则对数据集进行重复。
4.  **随机跳过**：在训练模式下，随机跳过部分数据以增加数据的多样性。
5.  **数据类型转换**：将数据集中所有整数转换为 `int32` 类型。
6.  **批量大小设置**：根据 `batch_size_means_tokens_param` 判断批量大小的含义，并根据 TPU 或 GPU 的不同进行相应处理。
7.  **数据过滤**：根据有效长度过滤数据。
8.  **批量处理**：根据不同的条件（如是否使用 TPU）进行批量处理和填充。
9.  **数据分块**：如果目标序列过长，则将批次分块，以便模型可以逐块处理。
10.  **输出准备**：准备输出格式，确保返回的格式适合后续处理。

#### **2. Shift right**

**路径：**tensor2tensor\layers\common_layers.py

```
def shift_right(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])[:, :-1, :, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :, :]
  return shifted_targets
 
def shift_right_3d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets
 
 
def shift_right_2d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0]])[:, :-1]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1]
  return shifted_targets
```

**函数功能**

在解码器的输入中，目标序列（例如翻译后的句子）会被 “右移” 一个位置。即，实际输入给解码器的序列是目标序列的前 n-1 个词汇，前面加上一个开始标记（通常是`<sos>`，Start of Sequence）。

**参数说明**

1.  **`x`:** 输入的张量，三个函数分别对应四维、三维、二维张量
2.  **`pad_value`:** 可选参数，用于填充的值。如果为 `None`，则使用零填充；如果提供了具体值，则使用该值进行填充。

**逻辑:**

*   如果 `pad_value` 为 `None`，则使用 `tf.pad` 函数将 `x` 的第二维（时间维度）向右填充一个位置，并去掉最后一个位置的元素。
*   如果 `pad_value` 不为 `None`，则使用 `tf.concat` 将 `pad_value` 和 `x` 进行拼接，然后去掉最后一个位置的元素。

**测试：**

```
x = [[1,2,3,4,5,6,7,8,9,10],
     [1,2,3,4,5,6,7,8,9,10]]
shift_right_x = shift_right_2d(x)
print(shift_right_x)
```

**结果：**

> tf.Tensor([[0 1 2 3 4 5 6 7 8 9]  
>                 [0 1 2 3 4 5 6 7 8 9]], shape=(2, 10), dtype=int32)

### 2. Embedding

**路径：**tensor2tensor\layers\common_layers.py

```
def embedding(x,
              vocab_size,
              dense_size,
              name=None,
              reuse=None,
              multiplier=1.0,
              symbol_dropout_rate=0.0,
              embedding_var=None,
              dtype=tf.float32):
  """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
  with tf.variable_scope(
      name, default_, values=[x], reuse=reuse, dtype=dtype):
    # 如果没有提供 embedding_var，创建一个新的嵌入矩阵变量。
    if embedding_var is None:
      embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
    # On the backwards pass, we want to convert the gradient from
    # an indexed-slices to a regular tensor before sending it back to the
    # parameter server. This avoids excess computation on the parameter server.
    # 在反向传播期间，将梯度从 indexed-slices 转换为常规张量。
    if not tf.executing_eagerly():
      embedding_var = convert_gradient_to_tensor(embedding_var)
    # 应用符号级别的丢弃。
    x = dropout_no_scaling(x, 1.0 - symbol_dropout_rate)
    # 根据输入 x 的索引从嵌入矩阵中收集对应的向量。
    emb_x = gather(embedding_var, x, dtype)
    # 如果 multiplier 不等于 1，将嵌入向量乘以 multiplier。
    if multiplier != 1.0:
      emb_x *= multiplier
    # 获取嵌入向量的静态形状。
    static_shape = emb_x.shape.as_list()
    # 如果维度少于 5，直接返回嵌入向量。
    if len(static_shape) < 5:
      return emb_x
    # 确保 static_shape 的长度为 5。
    assert len(static_shape) == 5
    # If we had an extra channel dimension, assume it's 1, i.e. shape[3] == 1.
    # 如果有一个额外的通道维度，假设它是 1，然后移除这个维度。
    return tf.squeeze(emb_x, 3)
```

**函数功能**

这个 `embedding` 函数的目的是将输入的整数索引（通常是词汇表中的词）转换为稠密的向量表示（嵌入向量）。该函数支持多种功能，如丢弃、乘法缩放等。下面是对该函数的详细解释。

函数参数

*   `x`: 输入的整数张量，通常是词汇表中词的索引，类型为 `int64`。
*   `vocab_size`: 词汇表的大小，表示可以嵌入的不同词的数量。
*   `dense_size`: 每个词的嵌入向量的维度。
*   `name`: 可选的变量作用域名称。
*   `reuse`: 可选的重用标志，指示是否重用已有的变量。
*   `multiplier`: 可选的缩放因子，默认值为 `1.0`，用于放大或缩小嵌入向量。
*   `symbol_dropout_rate`: 符号级别的丢弃率，默认值为 `0.0`，用于在训练时随机丢弃一些输入。
*   `embedding_var`: 可选的嵌入矩阵变量，如果提供则使用该变量而不是创建新的。
*   `dtype`: 数据类型，默认为 `tf.float32`。

**代码逻辑**

1.  **变量作用域**：使用 `tf.variable_scope` 创建一个变量作用域，以便于管理变量的命名和重用。
2.  **创建嵌入矩阵**：如果没有提供 `embedding_var`，则创建一个新的嵌入矩阵变量，形状为 `[vocab_size, dense_size]`。
3.  **梯度转换**：在非急切执行模式下，将嵌入变量的梯度从 `indexed-slices` 转换为常规张量，以减少参数服务器上的计算负担。
4.  **符号级别丢弃**：应用符号级别的丢弃，使用 `dropout_no_scaling` 函数来随机丢弃一些输入。
5.  **收集嵌入向量**：根据输入 `x` 的索引从嵌入矩阵中收集对应的嵌入向量，使用 `gather` 函数。
6.  **缩放嵌入向量**：如果 `multiplier` 不等于 `1.0`，则将嵌入向量乘以该因子。
7.  **返回嵌入向量**：
    *   获取嵌入向量的静态形状。
    *   如果维度少于 5，直接返回嵌入向量。
    *   如果有额外的通道维度（即形状长度为 5），则假设该维度为 1，并使用 `tf.squeeze` 移除该维度。

### 3. Position Encoding

在 Transformer 模型中，由于自注意力机制本身不包含任何顺序信息，因此需要额外的位置编码来提供这种信息。其中包含三种方式：

*   `timing`: 使用时间信号。
*   `timing_from_features`: 从特征中添加时间信号。
*   `emb`: 使用位置嵌入。

```
"""Adds a bunch of sinusoids of different frequencies to a Tensor.
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  expressed in terms of b, sin(a) and cos(a).
  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image
  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
"""
```

**翻译：**

> 将一组不同频率的正弦波添加到张量中。
> 
> 输入张量的每个通道都由一个位置维度中不同频率和相位的正弦波增加。这使得注意力可以学习使用绝对位置和相对位置。应将时间信号添加到查询和内存输入的一些前体中以引起注意。使用相对位置是可能的，因为 sin(a+b) 和 cos(a+b) 可以用 b、sin(a) 和 cos(a) 来表示。
> 
> x 是具有 n 个 “位置” 维度的张量，例如序列的一个维度或图像的两个维度。我们使用以 min_timescale 开头并以 max_timescale 结尾的时间尺度的几何序列。不同时间尺度的数量等于通道 // (n * 2)。对于每个时间尺度，我们生成两个正弦信号 sin(timestep/timescale) 和 cos(timestep/timescale)。所有这些正弦波都在通道维度上连接起来。

**路径：**tensor2tensor\layers\common_layers.py

```
def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16):
  """Create Tensor of sinusoids of different frequencies.
  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int
  Returns:
    Tensor of shape (length, 2*num_timescales)
  """
  # 将步长转换为浮点数。
  positions = to_float(tf.range(length))
   # 计算时间尺度增量。
  log_timescale_increment = (
      math.log(max_timescale / min_timescale) / (num_timescales - 1))
  # 计算每个时间尺度的倒数。
  inv_timescales = min_timescale * tf.exp(
      to_float(tf.range(num_timescales)) * -log_timescale_increment)
  # 扩展时间步长和倒时间尺度以进行矩阵乘法。
  scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
  # 通过计算正弦和余弦值来创建定时信号。
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
 
```

 **函数功能**

`get_timing_signal` 函数用于生成正弦和余弦函数的周期性信号，这些信号用于给序列模型提供时间信息

**关键步骤：**

1.  使用 `tf.range(length)` 创建一个从 0 到 `length-1` 的序列，表示序列中的每个位置。
2.  将这些位置转换为浮点数，以便进行后续的数学运算。
3.  计算时间尺度增量，这决定了正弦和余弦函数的频率间隔。
4.  计算每个时间尺度的倒数，这将用于确定正弦和余弦函数的周期。
5.  通过将位置与倒时间尺度相乘，创建一个扩展的时间维度，用于生成正弦和余弦值。
6.  使用 `tf.sin` 和 `tf.cos` 函数计算正弦和余弦值，并将它们在最后一个维度上连接起来，形成一个包含两种信号的 Tensor。

```
def add_timing_signal(x, min_timescale=1, max_timescale=1e4, num_timescales=16):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  Args:
    x: a Tensor with shape [?, length, ?, depth]
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int <= depth/2
  Returns:
    a Tensor the same shape as x.
  """
  length = shape_list(x)[1]
  depth = shape_list(x)[3]
  signal = get_timing_signal(length, min_timescale, max_timescale,
                             num_timescales)
  padded_signal = tf.pad(signal, [[0, 0], [0, depth - 2 * num_timescales]])
  return x + tf.reshape(padded_signal, [1, length, 1, depth])
```

 **函数功能**

`add_timing_signal` 函数的目的是向输入的张量 `x` 添加不同频率的正弦信号，以便于模型在处理序列数据时能够捕捉到位置信息

#### 参数说明

*   **`x`**: 输入的张量，形状为 `[?, length, ?, depth]`，其中 `length` 是序列长度，`depth` 是特征维度。
*   **`min_timescale`**: 最小时间尺度，控制正弦波的最低频率。
*   **`max_timescale`**: 最大时间尺度，控制正弦波的最高频率。
*   **`num_timescales`**: 时间尺度的数量，应该小于等于 `depth/2`，用于生成不同频率的正弦信号。

#### 返回值

*   返回一个与输入 `x` 形状相同的张量，经过加上定时信号的处理。

#### 关键步骤

1.  **获取输入的长度和深度**: 使用 `shape_list(x)` 函数来提取 `x` 的形状信息。
2.  **生成定时信号**: 调用 `get_timing_signal` 函数生成一个包含不同频率的正弦信号。
3.  **填充信号**: 使用 `tf.pad` 对信号进行填充，以确保其深度与输入 `x` 的深度一致。
4.  **加和操作**: 将填充后的信号与输入 `x` 相加，并通过 `tf.reshape` 调整信号的形状，以便进行广播。

### 4. transformer_prepare_encoder

**位置：**tensor2tensor\layers\transformer_layers.py

```
def transformer_prepare_encoder(inputs, target_space, hparams, features=None,
                                type_ids=None, num_types=None,
                                reuse_target_embedding=tf.AUTO_REUSE):
  """Prepare one shard of the model for the encoder.
  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.
    type_ids: optional, an int64 Tensor of shape [batch, length] that allows
      for adding type embeddings, similar to positional embeddings.
    num_types: optional, an int that decides the number of types in type_ids.
    reuse_target_embedding: option to reuse variable name in the case that
      symbol modalities are reused between inputs/targets.
  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  if features and "inputs_segmentation" in features:
    # Packed dataset.  Keep the examples from seeing each other.
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      encoder_self_attention_bias = (
          common_attention.attention_bias_same_segment(
              inputs_segmentation, inputs_segmentation))
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(targets_segmentation,
                                                     inputs_segmentation))
  else:
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      # Usual case - not a packed dataset.
      encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])
  if target_space is not None and hparams.get("use_target_space_embedding",
                                              True):
    # Append target_space_id embedding to inputs.
    emb_target_space = common_layers.embedding(
        target_space,
        32,
        ishape_static[-1],
        ,
        dtype=hparams.get("activation_dtype", "float32"),
        reuse=reuse_target_embedding)
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space
  if hparams.pos == "timing":
    if inputs_position is not None:
      encoder_input = common_attention.add_timing_signal_1d_given_position(
          encoder_input, inputs_position)
    else:
      encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  elif hparams.pos == "timing_from_features":
    encoder_input = common_attention.add_timing_signals_from_features(
        encoder_input, features, hparams.position_features)
  elif hparams.pos == "emb":
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "inputs_positional_embedding",
        inputs_position)
 
  # Add type embeddings
  if type_ids is not None:
    if not num_types:
      raise ValueError("Need to set num_types as well.")
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, num_types, "inputs_type_embedding", type_ids)
 
  encoder_self_attention_bias = common_layers.cast_like(
      encoder_self_attention_bias, encoder_input)
  encoder_decoder_attention_bias = common_layers.cast_like(
      encoder_decoder_attention_bias, encoder_input)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)
```

**作用：**为 Transformer 模型的编码器准备输入和自注意力偏置，考虑了因果性、填充、位置编码、目标空间嵌入和类型嵌入等多种因素，确保编码器在训练和推理时能够正确处理输入数据。

**函数参数**

*   `inputs`: 输入的张量，通常是经过嵌入层处理后的数据。
*   `target_space`: 目标空间的张量，可能用于生成目标空间的嵌入。
*   `hparams`: 超参数对象，包含模型运行所需的各种参数。
*   `features`: 可选的特征字典，包含额外的信息，特别是在使用 “打包” 数据集时。
*   `type_ids`: 可选的张量，形状为 `[batch, length]`，用于添加类型嵌入，类似于位置嵌入。
*   `num_types`: 可选的整数，表示 `type_ids` 中的类型数量。
*   `reuse_target_embedding`: 可选参数，决定是否重用目标嵌入的变量名称。

**返回值**

*   `encoder_input`: 编码器输入张量，经过处理后的输入。
*   `encoder_self_attention_bias`: 用于编码器自注意力的偏置张量。
*   `encoder_decoder_attention_bias`: 用于编码器 - 解码器注意力的偏置张量。

**函数逻辑**

**1. 初始化和处理 “打包” 数据集**

```
ishape_static = inputs.shape.as_list()
encoder_input = inputs
if features and "inputs_segmentation" in features:
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
        tf.logging.info("Using unidirectional encoder")
        encoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(
                common_layers.shape_list(inputs)[1]))
    else:
        encoder_self_attention_bias = (
            common_attention.attention_bias_same_segment(
                inputs_segmentation, inputs_segmentation))
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(targets_segmentation,
                                                     inputs_segmentation))
else:
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
        tf.logging.info("Using unidirectional encoder")
        encoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(
                common_layers.shape_list(inputs)[1]))
    else:
        encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
```

*   检查是否提供了特征和输入分段信息，以处理 “打包” 数据集。
*   根据是否启用单向编码器，选择不同的自注意力偏置计算方式。
*   如果不是 “打包” 数据集，则计算填充和忽略填充的偏置。

**2. 处理邻近偏置**

```
if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])
```

*   如果启用了邻近偏置，则添加相应的偏置。

**3. 添加目标空间嵌入**

```
if target_space is not None and hparams.get("use_target_space_embedding",
                                              True):
    emb_target_space = common_layers.embedding(
        target_space,
        32,
        ishape_static[-1],
        ,
        dtype=hparams.get("activation_dtype", "float32"),
        reuse=reuse_target_embedding)
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space
```

*   如果目标空间不为`None`且启用了目标空间嵌入，则计算并添加目标空间的嵌入。

**4. 添加位置编码**

```
if hparams.pos == "timing":
    if inputs_position is not None:
        encoder_input = common_attention.add_timing_signal_1d_given_position(
            encoder_input, inputs_position)
    else:
        encoder_input = common_attention.add_timing_signal_1d(encoder_input)
elif hparams.pos == "timing_from_features":
    encoder_input = common_attention.add_timing_signals_from_features(
        encoder_input, features, hparams.position_features)
elif hparams.pos == "emb":
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "inputs_positional_embedding",
        inputs_position)
```

*   根据不同的`hparams.pos`值，添加不同的位置信号或位置嵌入。

**5. 添加类型嵌入**

```
if type_ids is not None:
    if not num_types:
        raise ValueError("Need to set num_types as well.")
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, num_types, "inputs_type_embedding", type_ids)
```

*   如果提供了`type_ids`，则添加类型嵌入。

**6. 类型转换和返回结果**

```
encoder_self_attention_bias = common_layers.cast_like(
    encoder_self_attention_bias, encoder_input)
encoder_decoder_attention_bias = common_layers.cast_like(
    encoder_decoder_attention_bias, encoder_input)
return (encoder_input, encoder_self_attention_bias,
        encoder_decoder_attention_bias)
```

*   将自注意力偏置和编码器 - 解码器注意力偏置转换为与编码器输入相同的数据类型，并返回编码器输入、自注意力偏置和编码器 - 解码器注意力偏置。

三、编码器（Encoder）
--------------

路径：tensor2tensor\layers\transformer_layers.py

```
def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        ,
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        attn_bias_for_padding=None):
  """A stack of transformer layers.
  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convolutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    attn_bias_for_padding: Padded attention bias in case a unidirectional
      encoder is being used where future attention is masked.
  Returns:
    y: a Tensors
  """
  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })
 
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        attention_bias = attn_bias_for_padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):
          if layer < hparams.get("num_area_layers", 0):
            max_area_width = hparams.get("max_area_width", 1)
            max_area_height = hparams.get("max_area_height", 1)
            memory_height = hparams.get("memory_height", 1)
          else:
            max_area_width = 1
            max_area_height = 1
            memory_height = 1
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),
              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              hard_attention_k=hparams.get("hard_attention_k", 0),
              gumbel_noise_weight=hparams.get("gumbel_noise_weight", 0.0),
              max_area_width=max_area_width,
              max_area_height=max_area_height,
              memory_height=memory_height,
              area_key_mode=hparams.get("area_key_mode", "none"),
              area_value_mode=hparams.get("area_value_mode", "none"),
              training=(hparams.get("mode", tf_estimator.ModeKeys.TRAIN)
                        == tf_estimator.ModeKeys.TRAIN))
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              pad_remover,
              conv_padding="SAME",
              nonpadding_mask=nonpadding,
              losses=losses)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)
```

**函数功能**

`transformer_encoder` 函数实现了一个 Transformer 编码器的堆叠结构。它由多个编码器层组成，每个层包括自注意力机制和前馈神经网络。以下是对该函数的详细解释，包括参数、实现步骤及示例代码。

**函数参数**

*   `encoder_input`: 输入张量，通常是经过嵌入层处理后的输入序列。
*   `encoder_self_attention_bias`: 自注意力的偏置张量，用于控制注意力机制。
*   `hparams`: 模型的超参数，包含编码器层数、隐藏层大小、注意力头数等。
*   `name`: 可选的字符串，用于变量作用域的命名。
*   `nonpadding`: 可选的张量，表示哪些位置不是填充（padding）。形状为 `[batch_size, encoder_length]`。
*   `save_weights_to`: 可选的字典，用于保存注意力权重以便可视化。
*   `make_image_summary`: 是否生成注意力图像摘要。
*   `losses`: 可选的列表，用于附加额外的训练损失。
*   `attn_bias_for_padding`: 用于单向编码器的填充注意力偏置。

**代码逻辑**

1.  **初始化输入**：将 `encoder_input` 赋值给 `x`。
2.  **获取超参数**：从 `hparams` 中获取注意力丢弃的广播维度等信息。
3.  **设置变量作用域**：使用 `tf.variable_scope` 创建一个变量作用域。
4.  **处理填充信息**：
    *   如果提供了 `nonpadding`，则计算 `padding`。
    *   如果没有，使用 `encoder_self_attention_bias` 计算 `padding` 和 `nonpadding`。
5.  **创建填充移除器**：如果启用了填充移除器且没有使用 XLA 编译，创建 `PadRemover` 实例。
6.  **堆叠编码器层**：使用循环迭代创建多个编码器层：
    *   **自注意力层**：调用 `common_attention.multihead_attention` 进行自注意力计算。
    *   **前馈神经网络层**：调用 `transformer_ffn_layer` 进行前馈计算。
7.  **输出处理**：在最后返回经过预处理的输出。

四、解码前准备
-------

**路径：**tensor2tensor\models\transformer.py

```
def transformer_prepare_decoder(targets, hparams, features=None, pad=None):
  """Prepare one shard of the model for the decoder.
  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    pad: vector to use for padding when shifting targets right
  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  if hparams.causal_decoder_self_attention:
    # Causal attention.
    if hparams.prepend_mode == "prepend_inputs_full_attention":
      decoder_self_attention_bias = (
          common_attention.attention_bias_prepend_inputs_full_attention(
              common_attention.embedding_to_padding(targets)))
    else:
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(targets)[1]))
  else:
    # Full attention.
    decoder_padding = common_attention.embedding_to_padding(targets)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(decoder_padding))
 
  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets, pad)
  if hparams.pos == "timing":
    if targets_position is not None:
      decoder_input = common_attention.add_timing_signal_1d_given_position(
          decoder_input, targets_position)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  elif hparams.pos == "timing_from_features":
    decoder_input = common_attention.add_timing_signals_from_features(
        decoder_input, features, hparams.position_features)
  elif hparams.pos == "emb":
    decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "targets_positional_embedding",
        targets_position)
 
  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)
```

**函数功能**

`transformer_prepare_decoder` 函数用于为 Transformer 模型的解码器准备输入和自注意力偏置。它处理目标序列的填充、位置编码和注意力机制的设置。以下是对该函数的详细解释，包括参数、实现步骤及示例代码。

**函数参数**

*   `targets`: 输入张量，通常是目标序列（例如，翻译任务中的目标语言句子）。
*   `hparams`: 运行时的超参数，包含解码器的配置。
*   `features`: 可选的字典，包含额外的特征信息，特别是在处理 “打包” 数据集时需要。
*   `pad`: 用于右移目标序列时的填充值。

**代码逻辑**

1.  **处理自注意力偏置**：
    
    *   如果启用了因果自注意力（`causal_decoder_self_attention`），则根据 `prepend_mode` 设置自注意力偏置：
        *   使用 `attention_bias_prepend_inputs_full_attention` 进行全输入注意力。
        *   使用 `attention_bias_lower_triangle` 进行下三角注意力。
    *   如果没有启用因果自注意力，则使用 `attention_bias_ignore_padding` 来忽略填充部分。
2.  **处理打包数据集**：
    
    *   如果提供了 `features` 并且包含 `targets_segmentation`，则使用 `attention_bias_same_segment` 来确保不同示例之间不会相互影响。
3.  **添加邻近偏置**：
    
    *   如果启用了邻近偏置（`proximity_bias`），则使用 `attention_bias_proximal` 添加偏置。
4.  **右移目标序列**：
    
    *   使用 `common_layers.shift_right_3d` 将目标序列右移，以便在解码时使用。
5.  **添加位置编码**：
    
    *   根据超参数 `pos` 的设置，添加不同类型的位置编码：
        *   `timing`: 使用时间信号。
        *   `timing_from_features`: 从特征中添加时间信号。
        *   `emb`: 使用位置嵌入。
6.  **数据类型转换**：
    
    *   如果 `activation_dtype` 为 `bfloat16`，则将自注意力偏置转换为 `bfloat16` 类型。
7.  **返回结果**：
    
    *   返回解码器输入和自注意力偏置。

五、解码器（Decoder）
--------------

### 1. 单个解码器层

**路径：**tensor2tensor\models\transformer.py

```
def transformer_decoder_layer(decoder_input,
                              decoder_self_attention_bias,
                              layer_idx,
                              hparams,
                              encoder_output=None,
                              encoder_decoder_attention_bias=None,
                              cache=None,
                              decode_loop_step=None,
                              nonpadding=None,
                              save_weights_to=None,
                              make_image_summary=False,
                              losses=None,
                              layer_collection=None,
                              recurrent_memory_by_layer=None,
                              chunk_number=None):
  """A single transformer decoder layer."""
  x, layer_cache = transformer_self_attention_layer(
      decoder_input=decoder_input,
      decoder_self_attention_bias=decoder_self_attention_bias,
      layer_idx=layer_idx,
      hparams=hparams,
      encoder_output=encoder_output,
      encoder_decoder_attention_bias=encoder_decoder_attention_bias,
      cache=cache,
      decode_loop_step=decode_loop_step,
      save_weights_to=save_weights_to,
      make_image_summary=make_image_summary,
      layer_collection=layer_collection,
      recurrent_memory_by_layer=recurrent_memory_by_layer,
      chunk_number=chunk_number)
 
  layer = layer_idx
  layer_name = "layer_%d" % layer
  with tf.variable_scope(layer_name):
    with tf.variable_scope("ffn"):
      y = transformer_ffn_layer(
          common_layers.layer_preprocess(
              x, hparams, layer_collection=layer_collection),
          hparams,
          conv_padding="LEFT",
          nonpadding_mask=nonpadding,
          losses=losses,
          cache=layer_cache,
          decode_loop_step=decode_loop_step,
          layer_collection=layer_collection)
      x = common_layers.layer_postprocess(x, y, hparams)
      return x
```

**函数功能**

`transformer_decoder_layer` 函数实现了 Transformer 模型的单个解码器层。该函数结合了自注意力层和前馈神经网络层的功能，处理解码器输入并生成输出。以下是对该函数的详细解释，包括参数、实现步骤及示例代码。

**函数参数**

*   `decoder_input`: 解码器的输入张量，通常是经过位置编码的目标序列。
*   `decoder_self_attention_bias`: 自注意力的偏置张量，用于控制注意力机制。
*   `layer_idx`: 当前解码器层的索引。
*   `hparams`: 运行时的超参数，包含解码器层的配置。
*   `encoder_output`: 可选的编码器输出，通常用于交叉注意力。
*   `encoder_decoder_attention_bias`: 编码器到解码器的注意力偏置。
*   `cache`: 用于存储解码器的缓存，以便在解码时重用。
*   `decode_loop_step`: 当前解码步骤，用于循环解码。
*   `nonpadding`: 非填充的掩码，指示哪些位置是有效的。
*   `save_weights_to`: 可选参数，用于保存权重。
*   `make_image_summary`: 是否生成图像摘要。
*   `losses`: 记录损失的列表。
*   `layer_collection`: 用于管理层的集合。
*   `recurrent_memory_by_layer`: 每层的递归记忆。
*   `chunk_number`: 当前块的编号。

**代码逻辑**

1.  **自注意力层**：
    
    *   调用 `transformer_self_attention_layer` 函数，传入解码器输入和自注意力偏置，计算自注意力的输出 `x` 和层缓存 `layer_cache`。
2.  **前馈神经网络层**：
    
    *   使用 `tf.variable_scope` 创建一个命名空间，便于管理变量。
    *   调用 `transformer_ffn_layer` 函数，传入经过预处理的输入 `x`，计算前馈网络的输出 `y`。
    *   使用 `common_layers.layer_postprocess` 函数将前馈网络的输出与输入 `x` 结合，得到最终的输出。
3.  **返回结果**：
    
    *   返回解码器层的输出 `x`。

### 2. 解码器

**路径：**tensor2tensor\models\transformer.py

```
def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        ,
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        layer_collection=None,
                        recurrent_memory_by_layer=None,
                        chunk_number=None):
  """A stack of transformer layers.
  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the KFAC
      optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
  Returns:
    y: a Tensors
  """
  x = decoder_input
 
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      },
      hparams=hparams)
 
  with tf.variable_scope(name):
    for layer_idx in range(hparams.num_decoder_layers or
                           hparams.num_hidden_layers):
      x = transformer_decoder_layer(
          x,
          decoder_self_attention_bias,
          layer_idx,
          hparams,
          encoder_decoder_attention_bias=encoder_decoder_attention_bias,
          encoder_output=encoder_output,
          cache=cache,
          decode_loop_step=decode_loop_step,
          nonpadding=nonpadding,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          losses=losses,
          layer_collection=layer_collection,
          recurrent_memory_by_layer=recurrent_memory_by_layer,
          chunk_number=chunk_number
          )
 
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(
        x, hparams, layer_collection=layer_collection)
```

**函数功能**

`transformer_decoder` 函数实现了 Transformer 模型的解码器部分，负责将解码器输入通过多个解码器层进行处理，最终生成输出。

**函数参数**

*   `decoder_input`: 解码器的输入张量，通常是经过位置编码的目标序列。
*   `encoder_output`: 编码器的输出张量，提供上下文信息。
*   `decoder_self_attention_bias`: 自注意力的偏置张量，用于控制注意力机制。
*   `encoder_decoder_attention_bias`: 编码器到解码器的注意力偏置。
*   `hparams`: 运行时的超参数，包含解码器层的配置。
*   `cache`: 用于存储解码器的缓存，以便在解码时重用。
*   `decode_loop_step`: 当前解码步骤，用于循环解码。
*   `name`: 解码器的名称，用于变量作用域。
*   `nonpadding`: 非填充的掩码，指示哪些位置是有效的。
*   `save_weights_to`: 可选参数，用于保存权重。
*   `make_image_summary`: 是否生成图像摘要。
*   `losses`: 记录损失的列表。
*   `layer_collection`: 用于管理层的集合。
*   `recurrent_memory_by_layer`: 每层的递归记忆。
*   `chunk_number`: 当前块的编号。

**代码逻辑**

1.  **初始化输入**：
    
    *   将 `decoder_input` 赋值给 `x`，作为解码器的初始输入。
2.  **日志记录**：
    
    *   使用 `mlperf_log` 记录模型的超参数信息，例如隐藏层数量、注意力丢弃率等。
3.  **创建变量作用域**：
    
    *   使用 `tf.variable_scope` 创建一个命名空间，以便于管理变量。
4.  **循环处理解码器层**：
    
    *   根据超参数中的层数，循环调用 `transformer_decoder_layer` 函数，逐层处理解码器输入 `x`，并将输出更新为新的输入。
5.  **输出处理**：
    
    *   在所有解码器层处理完毕后，使用 `common_layers.layer_preprocess` 函数对最终输出进行预处理，返回解码器的输出。

六、Softmax
---------

### 1. Saturating Sigmoid

```
def saturating_sigmoid(x):
  """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
  with tf.name_scope("saturating_sigmoid", values=[x]):
    y = tf.sigmoid(x)
    return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))
```

`saturating_sigmoid` 函数实现了一种饱和的 sigmoid 函数，其公式为：𝑦=min(1.0,max(0.0,1.2⋅𝜎(𝑥)−0.1))，其中，𝜎(𝑥) 是标准的 sigmoid 函数。

**代码逻辑**

*   **计算标准 sigmoid**：使用 TensorFlow 的 `tf.sigmoid` 函数计算输入 `x` 的 sigmoid 值。
*   **调整输出**：将 sigmoid 值乘以 1.2，然后减去 0.1，最后使用 `tf.minimum` 和 `tf.maximum` 将结果限制在 [0, 1] 的范围内。

### 2. Hard Sigmoid

```
 
def hard_sigmoid(x, saturation_limit=0.9):
  saturation_cost = tf.reduce_mean(tf.nn.relu(tf.abs(x) - saturation_limit))
  x_shifted = 0.5 * x + 0.5
  return tf.minimum(1.0, tf.nn.relu(x_shifted)), saturation_cost
```

`hard_sigmoid` 函数实现了一种硬 sigmoid 函数，其输出在一定范围内是线性的，超出该范围后则饱和。

**代码逻辑**

*   **计算饱和成本**：计算输入 `x` 超出饱和限制的部分的平均值，使用 `tf.reduce_mean` 和 `tf.nn.relu` 来实现。
*   **平移输入**：将输入 `x` 平移，使其范围在 [0, 1] 之间。
*   **计算硬 sigmoid**：使用 `tf.nn.relu` 将平移后的值限制在 [0, 1] 之间。

七、transformer 类
---------------

**路径：**tensor2tensor\models\transformer.py

```
class Transformer(t2t_model.T2TModel):
  """Attention net.  See file docstring."""
 
  def __init__(self, *args, **kwargs):
    super(Transformer, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # For visualizing attention heads.
    self.recurrent_memory_by_layer = None  # Override to enable recurrent memory
    self._encoder_function = transformer_encoder
    self._decoder_function = transformer_decoder
    self._init_cache_fn = _init_transformer_cache
    self._prepare_encoder_fn = transformer_prepare_encoder
    self._prepare_decoder_fn = transformer_prepare_decoder
 
  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)
 
  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    """Decode Transformer outputs, see transformer_decode."""
    return transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
        **kwargs)
 
  def body(self, features):
    """Transformer main model_fn.
    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.
    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams
 
    losses = []
 
    if self.has_input:
      inputs = self._prepare_inputs_for_body(features)
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)
 
    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
        targets, hparams, features=features)
 
    # Not all subclasses of Transformer support keyword arguments related to
    # recurrent memory, so only pass these arguments if memory is enabled.
    decode_kwargs = {}
    if self.recurrent_memory_by_layer is not None:
      # TODO(kitaev): The chunk_number feature currently has the same shape as
      # "targets", but this is only for the purposes of sharing sharding code.
      # In fact every token within an example must have the same chunk number.
      chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
      chunk_number_each_example = chunk_number_each_token[:, 0]
      # Uncomment the code below to verify that tokens within a batch share the
      # same chunk number:
      # with tf.control_dependencies([
      #     tf.assert_equal(chunk_number_each_token,
      #                     chunk_number_each_example[:, None])
      # ]):
      #   chunk_number_each_example = tf.identity(chunk_number_each_example)
      decode_kwargs = dict(
          recurrent_memory_by_layer=self.recurrent_memory_by_layer,
          chunk_number=chunk_number_each_example,
          )
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses,
        **decode_kwargs
        )
    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}
 
    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret
 
  def _prepare_inputs_for_body(self, features):
    """Prepare inputs for body.
    Args:
      features: Map of string to model features. Should contain
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
    Returns:
      Inputs which will be passed to the model. [batch_size, input_length, 1,
          hidden_dim]
    """
    return features["inputs"]
 
  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.
    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool. Whether to build the inference graph for TPU.
    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    # For real-valued modalities use the slow decode path for now.
    if (self._target_modality_is_real or
        self._hparams.self_attention_type != "dot_product"):
      return super(Transformer, self)._greedy_infer(features, decode_length)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length)
      return self._fast_decode(features, decode_length)
 
  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.
    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.
    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    if (self._hparams.self_attention_type not in [
        "dot_product", "dot_product_relative"
    ]):
      # Caching is not guaranteed to work with attention types other than
      # dot_product and dot_product_relative.
      return self._beam_decode_slow(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length, beam_size,
                                     top_beams, alpha)
      return self._fast_decode(features, decode_length, beam_size, top_beams,
                               alpha)
 
  def _prepare_inputs_for_decode(self, features):
    """Prepare inputs for decoding.
    Args:
      features: A map of string to model features.
    Returns:
      Inputs after fixing shape and applying modality.
    """
    dp = self._data_parallelism
    hparams = self._hparams
    inputs = features["inputs"]
    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = common_layers.shape_list(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.modality["inputs"]
    input_vocab_size = self._problem_hparams.vocab_size["inputs"]
    if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
    modality_name = hparams.name.get("inputs",
                                     modalities.get_name(input_modality))(
                                         hparams, input_vocab_size)
    with tf.variable_scope(modality_name):
      bottom = hparams.bottom.get("inputs",
                                  modalities.get_bottom(input_modality))
      inputs = dp(bottom, inputs, hparams, input_vocab_size)
    return inputs
 
  def _fast_decode_tpu(self,
                       features,
                       decode_length,
                       beam_size=1,
                       top_beams=1,
                       alpha=1.0):
    """Fast decoding.
    Implements both greedy and beam search decoding on TPU, uses beam search
    iff beam_size > 1, otherwise beam search related arguments are ignored.
    Args:
      features: A map of string to model features.
      decode_length: An integer, how many additional timesteps to decode.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha,
        stronger the preference for longer translations.
    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }.
    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
 
    if self.has_input:
      inputs_shape = common_layers.shape_list(features["inputs"])
      if (target_modality == modalities.ModalityType.CLASS_LABEL or
          self._problem_hparams.get("regression_targets")):
        decode_length = 1
      else:
        decode_length = (
            inputs_shape[1] + features.get("decode_length", decode_length))
      batch_size = inputs_shape[0]
      inputs = self._prepare_inputs_for_decode(features)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None
 
      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]
 
    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "timing_from_features":
      positional_encoding = common_attention.add_timing_signals_from_features(
          tf.zeros([1, decode_length + 1, hparams.hidden_size]), features,
          hparams.position_features)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length + 1, hparams.hidden_size]),
          hparams.max_length, "body/targets_positional_embedding", None)
    else:
      positional_encoding = None
 
    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.
      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.
      Args:
        targets: A tensor, inputs ids to the decoder. [batch_size, 1].
        i: An integer, Step number of the decoding loop.
      Returns:
        A tensor, processed targets [batch_size, 1, hidden_dim].
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)
 
      # GO embeddings are all zero, this is because transformer_prepare_decoder
      # Shifts the targets along by one for the input which pads with zeros.
      # If the modality already maps GO to the zero embeddings this is not
      # needed.
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)
 
      if positional_encoding is not None:
        positional_encoding_shape = positional_encoding.shape.as_list()
        targets += tf.slice(
            positional_encoding, [0, i, 0],
            [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
      return targets
 
    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)
 
    def symbols_to_logits_tpu_fn(ids, i, cache):
      """Go from ids to logits for next symbol on TPU.
      Args:
        ids: A tensor, symbol IDs.
        i: An integer, step number of the decoding loop. Only used for inference
          on TPU.
        cache: A dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      Returns:
        ret: A tensor, computed logits.
        cache: A dict, containing tensors which are the results of previous
            attentions, used for fast decoding.
      """
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)
 
      bias_shape = decoder_self_attention_bias.shape.as_list()
      bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                      [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
 
      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            i,
            nonpadding=features_to_nonpadding(features, "targets"))
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets",
                              modalities.get_top(target_modality))
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]
 
      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]
 
        def forced_logits():
          return tf.one_hot(
              tf.tile(
                  tf.slice(partial_targets, [0, i],
                           [partial_targets.shape.as_list()[0], 1]),
                  [beam_size]), vocab_size, 0.0, -1e9)
 
        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache
 
    eos_id = self.get_decode_end_id() or beam_search.EOS_ID
    temperature = features.get("sampling_temp",
                               getattr(hparams, "sampling_temp", 0.0))
    top_k = features.get("sampling_keep_top_k",
                         getattr(hparams, "sampling_keep_top_k", -1))
 
    ret = fast_decode_tpu(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_tpu_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        init_cache_fn=self._init_cache_fn,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length,
        eos_id=eos_id,
        sampling_temperature=temperature,
        top_k=top_k)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret
 
  def get_decode_start_id(self):
    """Returns the id of the first decoder input symbol.
    The default case maps None to a vector of 0's for transformer. This method
    can be overridden to return a different id by a model wanting to use a
    different decoder start symbol. The id returned by this method is used to
    index the embedding matrix, and retrieve the vector that will be used as the
    first input to the decoder
    """
    return None
 
  def get_decode_end_id(self):
    """Returns the id of the output symbol that terminates decoding.
    This method can be overridden by a different model. The id returned by this
    method is used to check if the generation is complete during decoding.
    """
    return None
 
  def _fast_decode(self,
```

`Transformer` 类实现了一个完整的 Transformer 模型，包含编码、解码、推理等功能。

#### 1. 类的构造函数

```
def __init__(self, *args, **kwargs):
    super(Transformer, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # 用于可视化注意力头
    self.recurrent_memory_by_layer = None  # 允许重写以启用递归记忆
    self._encoder_function = transformer_encoder
    self._decoder_function = transformer_decoder
    self._init_cache_fn = _init_transformer_cache
    self._prepare_encoder_fn = transformer_prepare_encoder
    self._prepare_decoder_fn = transformer_prepare_decoder
```

*   **`attention_weights`**: 用于存储注意力权重，以便后续可视化。
*   **`recurrent_memory_by_layer`**: 用于支持递归记忆的功能。
*   **`_encoder_function` 和 `_decoder_function`**: 指定编码器和解码器的实现函数。
*   **`_init_cache_fn`**: 初始化缓存的函数。
*   **`_prepare_encoder_fn` 和 `_prepare_decoder_fn`**: 准备编码器和解码器输入的函数。

#### 2. 编码方法

```
def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        features=features, losses=losses,
        prepare_encoder_fn=self._prepare_encoder_fn)
```

*   **`encode`**: 该方法接受输入数据、目标空间、超参数等，调用 `transformer_encode` 函数进行编码，返回编码后的输出。

#### 3. 解码方法

```
def decode(self,
           decoder_input,
           encoder_output,
           encoder_decoder_attention_bias,
           decoder_self_attention_bias,
           hparams,
           cache=None,
           decode_loop_step=None,
           nonpadding=None,
           losses=None,
           **kwargs):
    """Decode Transformer outputs, see transformer_decode."""
    return transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
        **kwargs)
```

*   **`decode`**: 该方法用于解码过程，接受解码器输入、编码器输出及相关的注意力偏置，调用 `transformer_decode` 函数进行解码。

#### 4. 模型主体

```
def body(self, features):
    """Transformer main model_fn.
    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.
    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams
 
    losses = []
 
    if self.has_input:
      inputs = self._prepare_inputs_for_body(features)
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)
 
    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = self._prepare_decoder_fn(
        targets, hparams, features=features)
 
    # 处理递归记忆相关的参数
    decode_kwargs = {}
    if self.recurrent_memory_by_layer is not None:
      chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
      chunk_number_each_example = chunk_number_each_token[:, 0]
      decode_kwargs = dict(
          recurrent_memory_by_layer=self.recurrent_memory_by_layer,
          chunk_number=chunk_number_each_example,
      )
      
    # 解码过程
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses,
        **decode_kwargs
    )
    
    # 处理期望的注意力损失
    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}
 
    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret
```

*   **`body`**: 该方法是 Transformer 模型的主要函数，负责处理输入、编码、解码，并返回最终的解码输出。它还处理损失计算和形状调整。

#### 5. 推理方法

*   **贪婪推理**:

```
def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.
    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool. Whether to build the inference graph for TPU.
    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if (self._target_modality_is_real or
        self._hparams.self_attention_type != "dot_product"):
      return super(Transformer, self)._greedy_infer(features, decode_length)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length)
      return self._fast_decode(features, decode_length)
```

*   **束搜索推理**:

```
def _beam_decode(self,
                 features,
                 decode_length,
                 beam_size,
                 top_beams,
                 alpha,
                 use_tpu=False):
    """Beam search decoding.
    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.
    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    if (self._hparams.self_attention_type not in [
        "dot_product", "dot_product_relative"
    ]):
      return self._beam_decode_slow(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length, beam_size,
                                     top_beams, alpha)
      return self._fast_decode(features, decode_length, beam_size, top_beams,
                               alpha)
```

*   **推理方法**：提供了两种推理方式：贪婪推理和束搜索推理，分别用于不同的解码需求。