from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # 将任意形状的输入 x 变成二维矩阵 (N, D)：
    # - x.shape[0] 是 batch size N；
    # - -1 是 NumPy 的 reshape 语法：这一维由 NumPy 自动推断为 D=∏d_i。
    x_reshaped = x.reshape(x.shape[0], -1)

    # 全连接/仿射变换：out = xW + b
    # - x_reshaped.dot(w) 是矩阵乘法： (N, D) · (D, M) -> (N, M)
    # - + b 利用广播(broadcast)：(M,) 会按行扩展到 (N, M)。
    out = x_reshaped.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # 反向传播核心是把上游梯度 dout 通过仿射层的计算图“链式法则”传回去。
    # 设 x_reshaped = x.reshape(N, D)，前向 out = x_reshaped @ w + b。

    # dx：对输入 x 的梯度。
    # - dout.dot(w.T)：(N, M) · (M, D) -> (N, D)
    # - reshape(x.shape)：把 (N, D) 还原回原始输入形状 (N, d1, ..., dk)。
    dx = dout.dot(w.T).reshape(x.shape)

    # dw：对权重 w 的梯度。
    # - x.reshape(N, D)：与前向相同的展平；
    # - .T 转置得到 (D, N)；
    # - (D, N) · (N, M) -> (D, M)。
    dw = x.reshape(x.shape[0], -1).T.dot(dout)

    # db：对偏置 b 的梯度。
    # - b 在前向对每个样本都加了一次，所以对 batch 维求和：axis=0。
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # ReLU(x) = max(0, x)
    # np.maximum 会对两个数组逐元素取最大值：
    # - 这里 0 是标量，会广播到与 x 同形状；
    # - 输出 out 与 x 形状一致。
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    
    
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # ReLU 的导数：x>0 时为 1，否则为 0。
    # (x > 0) 产生布尔数组，NumPy 中在乘法里会自动转成 {0,1}。
    # 逐元素相乘实现“门控”：把对应于 x<=0 的上游梯度置 0。
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # 1) 计算当前 mini-batch 的均值：对 batch 维 N 求均值，保留特征维 D。
        #    axis=0 表示沿着第 0 维(N)聚合，得到 shape (D,)。
        sample_mean = np.mean(x, axis=0)

        # 2) 去中心化：x - mean
        #    这里 sample_mean 会广播到 (N, D)。
        xmu = x - sample_mean

        # 3) 计算方差：E[(x-mean)^2]
        #    用逐元素乘法 xmu * xmu 得到平方，再对 N 求均值。
        sample_var = np.mean(xmu * xmu, axis=0)

        # 4) 计算标准差：sqrt(var + eps)
        #    eps 防止方差为 0 导致除零。
        sqrtvar = np.sqrt(sample_var + eps)

        # 5) 计算标准差的倒数：1/sqrt(var+eps)
        #    用浮点 1.0 确保是浮点除法。
        invvar = 1.0 / sqrtvar

        # 6) 标准化：x_hat = (x - mean) / sqrt(var+eps)
        #    这里 invvar 会广播到 (N, D)。
        xhat = xmu * invvar

        # 7) 缩放 + 平移：out = gamma * x_hat + beta
        #    gamma/beta 形状 (D,) 会按列广播到 (N, D)。
        out = gamma * xhat + beta

        # 8) 更新滑动平均统计量（供 test 时使用）：
        #    momentum 越大越“信任”历史；(1-momentum) 权重给当前 batch。
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # 9) cache 保存反向传播所需中间变量（按你的 backward 推导来存）。
        cache = (xhat, gamma, xmu, invvar, sqrtvar, sample_var, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # test 模式不使用当前 batch 的统计量，而使用训练时维护的 running_mean/var。
        # 归一化：同样是 (x - mean) / sqrt(var+eps)。
        xhat = (x - running_mean) / np.sqrt(running_var + eps)

        # 再做缩放和平移（gamma/beta 依然按列广播）。
        out = gamma * xhat + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # cache 中的变量来自 batchnorm_forward(train) 的中间结果。
    xhat, gamma, xmu, invvar, sqrtvar, var, eps = cache

    # dout 的形状是 (N, D)：N 个样本，D 个特征。
    N, D = dout.shape

    # beta 的梯度：out = gamma*xhat + beta
    # 对每个特征维求和即可（beta 在 N 个样本上被重复加）。
    dbeta = np.sum(dout, axis=0)

    # gamma 的梯度：out 对 gamma 的偏导是 xhat，所以 sum(dout * xhat)。
    dgamma = np.sum(dout * xhat, axis=0)

    # 先把梯度传到 xhat：dxhat = dout * gamma（逐元素乘法 + 广播）。
    dxhat = dout * gamma

    # 方差 var 的梯度：
    # invvar = (var+eps)^(-1/2)，所以 d(invvar)/dvar = -1/2*(var+eps)^(-3/2)
    # 这里用 invvar**3 等价于 (var+eps)^(-3/2)。
    dvar = np.sum(dxhat * xmu, axis=0) * (-0.5) * (invvar ** 3)

    # 均值 mu 的梯度：来自两条路径
    # - xmu = x - mu（因此 dxmu 会贡献到 dmu）
    # - var = mean(xmu^2)（因此 dvar 也会通过 xmu 影响 mu）
    # np.mean(..., axis=0) 等价于 sum(...) / N。
    dmu = np.sum(dxhat * (-invvar), axis=0) + dvar * np.mean(-2.0 * xmu, axis=0)

    # 最终 dx：把三条梯度路径合并：
    # - 从 xhat 路径：dxhat * invvar
    # - 从 var 路径：dvar * 2*xmu/N
    # - 从 mu 路径：dmu / N
    dx = dxhat * invvar + dvar * (2.0 * xmu / N) + dmu / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # 这个 “alt” 版本把完整推导化简成一个更紧凑的公式。
    # 仍然复用同样的 cache。
    xhat, gamma, xmu, invvar, sqrtvar, var, eps = cache

    # N: batch 大小；D: 特征数。
    N, D = dout.shape

    # dbeta / dgamma 与标准版本一致。
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xhat, axis=0)

    # dxhat：先把梯度传到归一化后的 xhat。
    dxhat = dout * gamma

    # 化简后的 dx 公式（按列/特征维广播）：
    # - np.sum(dxhat, axis=0): 对 N 求和，得到每个特征的总梯度
    # - np.sum(dxhat * xhat, axis=0): 与 xhat 的相关项
    # 整体乘以 invvar 并除以 N。
    dx = (
        (1.0 / N)
        * invvar
        * (N * dxhat - np.sum(dxhat, axis=0) - xhat * np.sum(dxhat * xhat, axis=0))
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # LayerNorm 与 BatchNorm 的区别：
    # - BatchNorm：对每个特征维 D，在 batch 维 N 上做归一化。
    # - LayerNorm：对每个样本 n，在特征维 D 上做归一化。
    # 因此这里的 mean/var 都沿 axis=1（特征维）计算。

    # 1) 每个样本的均值：shape (N, 1)
    # keepdims=True 保留维度，方便后续与 (N, D) 做广播运算。
    sample_mean = np.mean(x, axis=1, keepdims=True)

    # 2) 去中心化：广播减法 (N, D) - (N, 1)
    xmu = x - sample_mean

    # 3) 方差：对特征维求均值，得到 shape (N, 1)
    sample_var = np.mean(xmu * xmu, axis=1, keepdims=True)

    # 4) 标准差与倒数（逐元素）：
    sqrtvar = np.sqrt(sample_var + eps)
    invvar = 1.0 / sqrtvar

    # 5) 归一化：广播乘法 (N, D) * (N, 1)
    xhat = xmu * invvar

    # 6) 按特征维缩放与平移：gamma/beta 形状 (D,) 会广播到 (N, D)
    out = gamma * xhat + beta

    # 7) cache：保存反向传播需要的中间量。
    cache = (xhat, gamma, xmu, invvar, sqrtvar, sample_var, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # LN 的反向可以直接复用 BN-alt 的结构，只是：
    # - 聚合维度从 N（batch 维）换成 D（特征维）；
    # - 因此 sum/mean 的 axis 也从 0 换成 1。
    xhat, gamma, xmu, invvar, sqrtvar, var, eps = cache

    # dout: (N, D)
    N, D = dout.shape

    # beta/gamma 的梯度仍然是对 batch 维求和。
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * xhat, axis=0)

    # 传回到 xhat。
    dxhat = dout * gamma

    # 化简后的 dx（沿特征维 axis=1 聚合）：
    # - np.sum(dxhat, axis=1, keepdims=True)：每个样本在 D 维上的梯度和
    # - np.sum(dxhat * xhat, axis=1, keepdims=True)：与 xhat 的相关项
    dx = (
      (1.0 / D)
      * invvar
      * (
        D * dxhat
        - np.sum(dxhat, axis=1, keepdims=True)
        - xhat * np.sum(dxhat * xhat, axis=1, keepdims=True)
      )
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # Inverted Dropout（倒置 dropout）：
        # - 训练时把保留下来的神经元按 1/p 放大，这样测试时就不需要再缩放；
        # - 这样保证 E[out] = x（期望不变）。

        # np.random.rand(*x.shape) 生成与 x 同形状的 [0,1) 均匀随机数。
        # “*x.shape” 是 Python 的参数解包语法：把形状元组拆成多个位置参数。
        rand = np.random.rand(*x.shape)

        # rand < p 得到布尔 mask（True 表示保留），再除以 p 变成 {0, 1/p} 的缩放 mask。
        # 这里的 "/ p" 是 inverted dropout 的关键：训练时做缩放。
        mask = (rand < p) / p

        # 逐元素乘法：把被丢弃的位置置 0，并对保留的位置乘以 1/p。
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # 测试时不做 dropout：直接原样通过。
        # （因为 inverted dropout 已经在训练时把期望校准过了。）
        out = x
        
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
      # 反向传播同样是逐元素乘：
      # - 对被丢弃的神经元（mask=0）梯度为 0；
      # - 对保留的神经元（mask=1/p）梯度按 1/p 放大，与前向缩放一致。
      dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
      # 测试时前向是恒等映射，因此反向梯度也原样传递。
      dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # 从超参数字典里取出 stride / pad。
    stride = conv_param["stride"]  # 卷积窗口每次滑动的步长。
    pad = conv_param["pad"]  # 在输入高宽两侧补零的像素数。

    # 解析输入/卷积核的形状。
    N, C, H, W = x.shape  # 输入张量形状：批大小、通道数、高、宽。
    F, _, HH, WW = w.shape  # 卷积核形状：(滤波器数F, 通道C, 高HH, 宽WW)。

    # 根据题目给的公式计算输出空间尺寸。
    # 使用整除 "//"：这里默认输入满足能整除的约束。
    H_out = 1 + (H + 2 * pad - HH) // stride  # 输出特征图高度 H'。
    W_out = 1 + (W + 2 * pad - WW) // stride  # 输出特征图宽度 W'。

    # 对输入做零填充：np.pad 的 pad_width 需要为每个维度给 (before, after)。
    # 这里只在 H/W 维度补 (pad, pad)，N/C 维度补 (0,0)。
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant"
    )

    # 初始化输出数组：shape (N, F, H_out, W_out)
    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)

    # 四重循环实现“朴素卷积”：
    # - 遍历样本 n
    # - 遍历滤波器 f（输出通道）
    # - 遍历输出空间位置 (i, j)
    for n in range(N):  # 遍历每个样本。
        for f in range(F):  # 对当前样本应用第 f 个滤波器。
            for i in range(H_out):  # 遍历输出高度方向位置。
                # 当前输出 i 对应输入窗口的起始行索引（步长为 stride）。
                hs = i * stride
                for j in range(W_out):  # 遍历输出宽度方向位置。
                    # 当前输出 j 对应输入窗口的起始列索引。
                    ws = j * stride

                    # 取出局部窗口：shape (C, HH, WW)
                    # 这里的切片语法 a[b:c] 是左闭右开区间。
                    window = x_padded[n, :, hs : hs + HH, ws : ws + WW]

                    # 逐元素乘法 window * w[f]：两者 shape 相同 (C, HH, WW)
                    # np.sum(...) 把三维求和得到标量，再加上对应偏置 b[f]。
                    out[n, f, i, j] = np.sum(window * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # cache 里是 forward 保存的输入/参数。
    x, w, b, conv_param = cache

    # 取出 stride / pad。
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    # 解析维度。
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    # dout: (N, F, H_out, W_out)
    _, _, H_out, W_out = dout.shape

    # forward 里对 x 做了 padding；反向也需要在 padded 空间里累加 dx。
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant"
    )
    dx_padded = np.zeros_like(x_padded)  # 在 padded 空间累积 dx。
    dw = np.zeros_like(w)  # dw 与 w 同形状。
    db = np.zeros_like(b)  # db 与 b 同形状。

    # b[f] 在 out 的所有空间位置 (i,j) 和所有样本 n 上都被加了一次。
    # 所以对 dout 的 (N, H_out, W_out) 维求和。
    db = np.sum(dout, axis=(0, 2, 3))

    # 朴素反向：与 forward 的四重循环对齐。
    # - dw[f] 累加：dout[n,f,i,j] * window
    # - dx_padded 的对应窗口累加：dout[n,f,i,j] * w[f]
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                hs = i * stride
                for j in range(W_out):
                    ws = j * stride
                    window = x_padded[n, :, hs : hs + HH, ws : ws + WW]

                    # dw：标量 dout 乘以对应输入窗口，逐元素累加到第 f 个卷积核。
                    dw[f] += dout[n, f, i, j] * window

                    # dx：标量 dout 乘以卷积核权重，逐元素累加到输入梯度的对应窗口。
                    dx_padded[n, :, hs : hs + HH, ws : ws + WW] += (
                        dout[n, f, i, j] * w[f]
                    )

    # 去掉 padding 部分，得到 dx 的原始输入形状 (N, C, H, W)。
    dx = dx_padded[:, :, pad : pad + H, pad : pad + W]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # 取出池化窗口大小与步长。
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    # 输入形状 (N, C, H, W)
    N, C, H, W = x.shape

    # 输出空间尺寸：不做 padding，所以使用 (H - pool_height) / stride。
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # 初始化输出：每个通道单独池化，不改变通道数。
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

    # 朴素池化：遍历样本、通道、输出空间位置。
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                hs = i * stride  # 窗口起始行。
                for j in range(W_out):
                    ws = j * stride  # 窗口起始列。

                    # 取池化窗口：shape (pool_height, pool_width)
                    window = x[n, c, hs : hs + pool_height, ws : ws + pool_width]

                    # 最大池化就是取窗口中的最大值。
                    out[n, c, i, j] = np.max(window)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # cache 中保存了前向的输入 x 以及池化参数。
    x, pool_param = cache
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    # 解析维度。
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    # 初始化输入梯度。
    dx = np.zeros_like(x)

    # 反向传播：最大池化只把梯度传给“取得最大值”的那个位置。
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                hs = i * stride
                for j in range(W_out):
                    ws = j * stride

                    # 前向时的池化窗口。
                    window = x[n, c, hs : hs + pool_height, ws : ws + pool_width]

                    # 找到窗口最大值。
                    m = np.max(window)

                    # mask 标记最大值位置：布尔数组会在乘法里转成 {0,1}。
                    mask = (window == m)

                    # 把对应 dout 分配到最大值位置；其余位置加 0。
                    dx[n, c, hs : hs + pool_height, ws : ws + pool_width] += (
                        dout[n, c, i, j] * mask
                    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Spatial BatchNorm 的目标：对每个通道 c，统计量在 (N, H, W) 上计算：
    #   mean_c = mean(x[:, c, :, :])，var_c = var(x[:, c, :, :])
    # 也就是把 N 张图、每张图的 HxW 个位置都当成“样本”，对通道维 C 做普通 BN。
    #
    # 关键做法：把输入从 (N, C, H, W) 变形成 (N*H*W, C)，然后复用你写的 batchnorm_forward。
    N, C, H, W = x.shape

    # x.transpose(0, 2, 3, 1)：重排轴顺序，把通道 C 放到最后
    #   (N, C, H, W) -> (N, H, W, C)
    # 这样每个空间位置 (n, h, w) 对应一个长度为 C 的“特征向量”。
    # 注意：transpose 通常返回“视图(view)”，不复制数据，只改变索引方式。
    x_nhwc = x.transpose(0, 2, 3, 1)

    # reshape(-1, C)：把 (N, H, W) 三个维度压平为一个维度 N*H*W。
    # 其中 -1 是 NumPy 的语法，表示“这一维由 NumPy 自动推断”。
    # 得到 (N*H*W, C)，每一行就是一个样本，列对应通道。
    x_reshaped = x_nhwc.reshape(-1, C)

    # 复用 vanilla BN：它会对形状 (M, C) 的输入按列(特征维)做归一化，
    # 这里的 M=N*H*W，所以等价于对每个通道在所有 (n,h,w) 上做 BN。
    # gamma/beta 形状是 (C,)，会按列广播(broadcast)到 (M, C)。
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)

    # 把输出从 (N*H*W, C) 还原回 (N, C, H, W)：
    #   (N*H*W, C) -> (N, H, W, C) -> (N, C, H, W)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # backward 的思路与 forward 完全对称：
    # 先把 dout 从 (N, C, H, W) 变成 (N*H*W, C)，复用 batchnorm_backward，
    # 再把 dx reshape 回 (N, C, H, W)。
    N, C, H, W = dout.shape

    # 同样先变成 NHWC 再压平：
    #   (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)

    # batchnorm_backward 会返回：
    #   dx: (N*H*W, C)
    #   dgamma/dbeta: (C,)
    dx, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)

    # 把 dx 还原到卷积特征图形状：
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # GroupNorm：把通道 C 分成 G 组，每组大小为 C//G；
    # 对每个样本 n、每组 g，在该组的 (C//G, H, W) 上做归一化。
    N, C, H, W = x.shape

    # reshape 把通道拆成 (G, C//G)：
    # (N, C, H, W) -> (N, G, C//G, H, W)
    # 这样每个 group 就是一块连续通道。
    x_group = x.reshape(N, G, C // G, H, W)

    # 计算每组的均值/方差：沿着组内的通道与空间维度求统计量。
    # axis=(2,3,4) 表示对 (C//G, H, W) 求均值/方差；
    # keepdims=True 保持维度，便于后续广播。
    mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_group, axis=(2, 3, 4), keepdims=True)

    # 标准差与倒数。
    sqrtvar = np.sqrt(var + eps)
    invvar = 1.0 / sqrtvar

    # 归一化：逐元素 (x - mean) * inv_std
    xhat_group = (x_group - mean) * invvar

    # 把分组后的结果还原回原形状 (N, C, H, W)。
    xhat = xhat_group.reshape(N, C, H, W)

    # 缩放和平移：gamma/beta 的形状 (1, C, 1, 1) 会按 N/H/W 广播。
    out = gamma * xhat + beta

    # cache 保存反向传播需要的量。
    cache = (G, xhat_group, invvar, gamma)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # cache 取出 forward 中保存的中间量。
    G, xhat_group, invvar, gamma = cache

    # dout: (N, C, H, W)
    N, C, H, W = dout.shape
    group_size = C // G  # 每组通道数。
    M = group_size * H * W  # 每组内参与归一化的元素总数。

    # beta/gamma 的梯度：对 (N, H, W) 维求和。
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(
      dout * (xhat_group.reshape(N, C, H, W)), axis=(0, 2, 3), keepdims=True
    )

    # 先把梯度传到 xhat。
    dxhat = dout * gamma

    # 重新分组，方便沿组内维度做求和。
    dxhat_group = dxhat.reshape(N, G, group_size, H, W)

    # LN/IN/GN 的“化简反向”通式：
    # dx = (1/M) * inv_std * (M*dxhat - sum(dxhat) - xhat*sum(dxhat*xhat))
    # 这里 sum(...) 都是在每个 group 内沿 (2,3,4) 聚合。
    sum1 = np.sum(dxhat_group, axis=(2, 3, 4), keepdims=True)
    sum2 = np.sum(dxhat_group * xhat_group, axis=(2, 3, 4), keepdims=True)
    dx_group = (1.0 / M) * invvar * (M * dxhat_group - sum1 - xhat_group * sum2)

    # 把分组后的 dx 还原回 (N, C, H, W)。
    dx = dx_group.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # N: batch 大小。
    N = x.shape[0]

    # 取出每个样本的正确类别分数：
    # - np.arange(N) 生成 [0..N-1] 作为行索引；
    # - y 作为列索引；
    # 高级索引会得到 shape (N,)。
    correct_class_scores = x[np.arange(N), y]

    # 计算所有类别的 hinge margins：max(0, s_j - s_y + 1)
    # correct_class_scores[:, None] 把 (N,) 变成 (N,1)，
    # 通过广播与 (N,C) 的 x 做减法。
    margins = np.maximum(0.0, x - correct_class_scores[:, None] + 1.0)

    # 正确类别的 margin 不计入损失，强制置 0。
    margins[np.arange(N), y] = 0.0

    # 损失是所有 margins 的平均。
    loss = np.sum(margins) / N

    # dx：对 x 的梯度。
    # 规则：若 margin>0，则对该类别分数的导数为 1；
    # 正确类别分数的导数为 -(#positive margins)。
    dx = np.zeros_like(x)
    positive = margins > 0  # 布尔掩码。
    dx[positive] = 1.0
    row_sum = np.sum(dx, axis=1)  # 每个样本正 margin 的个数。
    dx[np.arange(N), y] -= row_sum
    dx /= N  # 平均到 batch。

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    # 数值稳定性：softmax 对 logit 平移不变。
    # 减去每行最大值，避免 exp 溢出。
    shifted_logits = x - np.max(x, axis=1, keepdims=True)

    # exp(logits)：逐元素指数。
    exp_shifted = np.exp(shifted_logits)

    # Z：每行归一化常数（分母），shape (N,1) 便于广播。
    Z = np.sum(exp_shifted, axis=1, keepdims=True)

    # softmax 概率：逐元素除法 + 广播。
    probs = exp_shifted / Z

    N = x.shape[0]

    # 交叉熵损失：-log p(correct)
    # 这里用 shifted_logits 与 log(Z) 组合：
    # log p_y = shifted_logits_y - log(Z)
    # Z.squeeze() 把 (N,1) 压成 (N,) 便于与高级索引结果对齐。
    correct_scores = shifted_logits[np.arange(N), y]
    loss = -np.sum(correct_scores - np.log(Z.squeeze())) / N

    # dx：softmax + cross-entropy 的标准梯度：
    # dx = probs；对正确类减 1；最后除以 N。
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
