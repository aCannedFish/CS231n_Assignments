import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    
    z_i_norm = torch.linalg.norm(z_i)
    z_j_norm = torch.linalg.norm(z_j)
    norm_dot_product = torch.dot(z_i, z_j) / (z_i_norm * z_j_norm)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        denom_k = 0
        denom_k_n = 0
        for i in range(2 * N):
            if i != k:
                denom_k += torch.exp(sim(z_k, out[i]) / tau)
            if i != k + N:
                denom_k_n += torch.exp(sim(z_k_N, out[i]) / tau)

        l_k_k_n = -torch.log(torch.exp(sim(z_k, z_k_N) / tau) / denom_k)
        l_k_n_k = -torch.log(torch.exp(sim(z_k_N, z_k) / tau) / denom_k_n)
        total_loss += l_k_k_n + l_k_n_k
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################
    pos_pairs = torch.sum(out_left * out_right, dim=1, keepdim=True) / (
        torch.linalg.norm(out_left, dim=1, keepdim=True)
        * torch.linalg.norm(out_right, dim=1, keepdim=True)
    )
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################
    norm_out = torch.linalg.norm(out, dim=1, keepdim=True)
    sim_matrix = torch.mm(out, out.t()) / torch.mm(norm_out, norm_out.t())

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################
    
    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    # `torch.exp(...)` 会对张量中的每个元素逐元素应用指数函数。
    # 这里先把相似度矩阵 `sim_matrix` 按温度参数 `tau` 缩放，再计算 exp(sim / tau)，
    # 得到论文分母中每一项对应的指数值。
    # 因为 `sim_matrix` 的形状是 [2N, 2N]，所以 `exponential` 的形状也保持为 [2N, 2N]。
    exponential = torch.exp(sim_matrix / tau)
    
    # This binary mask zeros out terms where k=i.
    # `torch.ones_like(exponential)` 生成一个和 `exponential` 同形状的全 1 张量。
    # `torch.eye(2 * N, ...)` 生成一个 [2N, 2N] 的单位矩阵，对角线为 1，其余位置为 0。
    # 两者相减后，对角线位置变成 0，非对角线位置保持 1。
    # 最后的 `.bool()` 把数值张量转换成布尔张量：0 -> False，1 -> True。
    # 因而 `mask` 是一个“去掉对角线”的布尔掩码，用来排除 k=i 的自相似项。
    mask = (torch.ones_like(exponential) - torch.eye(2 * N, device=exponential.device)).bool()
    
    # We apply the binary mask.
    # `masked_select(mask)` 会按照布尔掩码只保留 `mask == True` 的元素，并返回一维张量。
    # 这里等价于从每一行里移除对角线上的那个元素，也就是去掉样本与自身的相似度项。
    # 随后的 `.view(2 * N, -1)` 把结果重新 reshape 成 [2N, 2N-1]：
    # - 第一维固定为 2N，表示每个增强样本各占一行；
    # - 第二维用 `-1` 让 PyTorch 自动推断为 2N-1。
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    # `torch.sum(..., dim=1, keepdim=True)` 表示沿着第 1 维（按行）求和。
    # 这里会把每个样本对应的所有“非自身”指数项加起来，得到损失分母。
    # `keepdim=True` 会保留这一维，因此输出形状是 [2N, 1]，便于后面与分子按元素相除。
    denom = torch.sum(exponential, dim=1, keepdim=True)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().
    # `sim_positive_pairs(out_left, out_right)` 会按行计算每个正样本对的余弦相似度，
    # 返回长度为 N 的张量，其中第 k 个元素对应 `(out_left[k], out_right[k])`。
    pos_pairs = sim_positive_pairs(out_left, out_right)
    # 在总损失中，每个正样本对会出现两次：
    # - 一次是 l(k, k+N)
    # - 一次是 l(k+N, k)
    # 因此用 `torch.cat([pos_pairs, pos_pairs], dim=0)` 在第 0 维把它复制并拼接，
    # 得到长度为 2N 的张量。
    # 再通过 `.view(2 * N, 1)` reshape 成列向量，形状与 `denom` 对齐。
    pos_pairs = torch.cat([pos_pairs, pos_pairs], dim=0).view(2 * N, 1)
    
    # Step 3: Compute the numerator value for all augmented samples.
    # 分子就是正样本对相似度对应的 exp(sim_pos / tau)。
    # 这里同样使用 `torch.exp` 做逐元素指数运算，输出形状仍为 [2N, 1]。
    numerator = torch.exp(pos_pairs / tau)
    
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    # `numerator / denom` 是按元素相除，得到每个样本对应的 softmax 概率项。
    # `torch.log(...)` 逐元素取对数；前面的负号实现公式中的 `-log(...)`。
    # `torch.sum(...)` 把 2N 个方向上的损失全部加起来。
    # 最后除以 `(2 * N)`，得到所有增强样本上的平均损失。
    loss = torch.sum(-torch.log(numerator / denom)) / (2 * N)
    
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    
    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))