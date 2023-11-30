"""
This .py file comes from https://github.com/JeffffffFu/Awesome-Differential-Privacy-and-Meachine-Learning
Author: Jeff, his home page https://space.bilibili.com/80356866/video
I am Xinpeng Ling, Email:  xpling@stu.ecnu.edu.cn
Home page: https://space.bilibili.com/3461572290677609
I migrated some utils needed in dp to this file.
"""

import numpy as np
import math
from scipy import special

"""---------------------引入Jeff库中data.util.compute_coordinates的cartesian_to_polar,
    polar_to_cartesian, vector_to_matrix, cartesian_add_noise, devide_epslion及相关函数----------------------------------------"""


def cartesian_to_polar(x):
    r = np.linalg.norm(x)
    theta = np.arccos(x[0] / r)  # 值域范围是[0,pi]
    phi = [1. for i in range(len(x) - 1)]
    for i in range(len(phi)):
        phi[i] = np.arctan2(x[i + 1], x[0])  # 以X[0]为标准计算反正切值,#值域范围是[-pi,pi]
    return np.concatenate(([r, theta], phi))


def polar_to_cartesian(p):
    r = p[0]
    theta = p[1]
    phi = p[2:]
    x = [1. for i in range(len(phi) + 1)]
    x[0] = r * np.cos(theta)  # 用这个求回X[0]没有问题
    for i in range(len(phi)):
        x[i + 1] = x[0] * np.tan(phi[i])
    for j in range(len(x)):
        x[j] = round(x[j], 4)  # 保留小数点后四位
    return x


def vector_to_matrix(vector, shape):
    shape = tuple(shape)
    if len(shape) == 0 or np.prod(shape) != len(vector):
        raise ValueError("Invalid input dimensions")
    matrix = np.zeros(shape)
    strides = [np.prod(shape[i + 1:]) for i in range(len(shape) - 1)] + [1]
    for i in range(len(vector)):
        index = [0] * len(shape)
        for j in range(len(shape)):
            index[j] = (i // strides[j]) % shape[j]
        matrix[tuple(index)] = vector[i]
    return matrix


# 加噪
def cartesian_add_noise(p, sigma1, C1, sigma2):
    # print("sigma1:{}".format(sigma1)+"| sigma2:{}".format(sigma2))
    r = p[0]

    # 对极值加噪，因为对梯度裁剪的时候其实就已经裁剪了
    r += C1 * sigma1 * np.random.normal(0, 1)

    theta = p[1:]  # 默认在-pi到pi之间，也就是2*pi
    theta += 2 * math.pi * sigma2 * np.random.normal(0, 1)

    return np.concatenate(([r], theta))


# 划分eps
def devide_epslion(sigma, q, n):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
    eps, opt_order = apply_dp_sgd_analysis(q, sigma, 1, orders, 10 ** (-5))
    # print("初始状态下每个梯度元素的eps:",eps)
    eps_sum = n * eps
    # print("LDP定义下的eps_sum:", eps_sum)
    eps1 = eps_sum * 0.000001
    # print("分给极值的eps1:",eps1)
    eps2 = eps_sum - eps1
    #  print("分给每一个角度的eps2:",eps2)
    sigma1 = get_noise_multiplier(target_epsilon=eps1, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    sigma2 = get_noise_multiplier(target_epsilon=eps2, target_delta=1e-5, sample_rate=512 / 60000, steps=1,
                                  alphas=orders)
    return sigma1, sigma2


def get_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        sample_rate: float,
        steps: int,
        alphas,
        epsilon_tolerance: float = 0.01,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    sigma_low, sigma_high = 0, 1000  # 从0-10进行搜索，一般的sigma设置也不会超过这个范围。其实从0-5就可以了我觉得。

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma_high, steps, alphas, target_delta)

    if eps_high > target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到100")

    # 下面是折半搜索，直到找到满足这个eps容忍度的sigma_high,sigma是从大到小搜索，即eps从小到大逼近
    while target_epsilon - eps_high > epsilon_tolerance:  # 我们希望当目前eps减去当前计算出来的eps小于容忍度，也就是计算出来的eps非常接近于目标eps
        sigma = (sigma_low + sigma_high) / 2

        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas, target_delta)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return round(sigma_high, 2)


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters.
    Args:
      n: Number of examples in the training data. 训练集样本总数
      batch_size: Batch size used in training. 一批采样的样本数
      noise_multiplier: Noise multiplier used in training. 噪声系数
      epochs: Number of epochs in training. 本地迭代轮次（还没有算上本地一次迭代中的多个batch迭代）
      delta: Value of delta for which to compute epsilon.
      S:sensitivity      这个原本的库是没有的
    Returns:
      Value of epsilon corresponding to input hyperparameters.  返回epsilon
    """
    q = batch_size / n  # q - the sampling ratio.          这里采样率=采样样本数/总样本数
    if q > 1:
        print('n must be larger than the batch size.')
    orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda

    # 总的steps应该是本地的epochs数*每次本地的batch迭代数（n/batch_size），因为每次batch迭代进行梯度下降
    # 由此可见，在做dataloder的时候，它也是按照这个规则去组装数据了，也就是batch_size为采样数，而每个dataloder会放n/batch_size个batch,注意batch之间应该是都要进行有放回的采样
    steps = int(math.ceil(epochs * (n / batch_size)))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


# 这个函数会调用计算RDP，和RDP转DP两个函数,如果只想传参steps，可以直接用这个函数而不用上面那个
def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    # compute_rdp requires that sigma be the ratio of the standard deviation of
    # the Gaussian noise to the l2-sensitivity of the function to which it is
    # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multilpier
    # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
    rdp = compute_rdp(q, sigma, steps, orders)  # 先算RDP、也就是RDP定义下总的隐私损失alpha，如果根据RDP算的话，可能会依据RDP的文章

    eps, opt_order = compute_eps(orders, rdp, delta)  # 再根据RDP转换为对应的最佳eps和lamda

    # print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
    #      ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
    # print('differential privacy with eps = {:.3g} and delta = {}.'.format(
    #   eps, delta))
    # print('The optimal RDP order is {}.'.format(opt_order))

    # if opt_order == max(orders) or opt_order == min(orders):            #这个是一个提示可以忽略，主要告诉我们可以扩展我们的orders的范围
    #   print('The privacy estimate is likely to be improved by expanding '
    #         'the set of orders.')

    return eps, opt_order


def compute_rdp(q, noise_multiplier, steps, orders):
    """Computes RDP of the Sampled Gaussian Mechanism.
    Args:
      q: The sampling rate.
      noise_multiplier: The ratio of the standard deviation of the Gaussian noise    STD标准差，敏感度应该包含在这里面了
        to the l2-sensitivity of the function to which it is added.
      steps: The number of steps.
      orders: An array (or a scalar) of RDP orders.
    Returns:
      The RDPs at all orders. Can be `np.inf`.
    """
    if np.isscalar(orders):  # 判断orders是不是标量类型，判断是否为一个数字还是一组list，只有一个数字走这个
        rdp = _compute_rdp(q, noise_multiplier, orders)  # 这里是具体计算采样下的RDP隐私损失的
    else:  # 如果是一个list，走的是这个函数，一般走这个
        rdp = np.array(
            [_compute_rdp(q, noise_multiplier, order) for order in orders])

    return rdp * steps  # 这里直接乘以总的迭代次数即可


def compute_eps(orders, rdp, delta):
    """Compute epsilon given a list of RDP values and target delta.
    Args:
      orders: An array (or a scalar) of orders.
      rdp: A list (or a scalar) of RDP guarantees.
      delta: The target delta.
    Returns:
      Pair of (eps, optimal_order).
    Raises:
      ValueError: If input is malformed.
    """
    orders_vec = np.atleast_1d(orders)  # 输入转换为至少一维的数组
    rdp_vec = np.atleast_1d(rdp)

    if delta <= 0:  # delat不能小于等于0
        raise ValueError("Privacy failure probability bound delta must be >0.")
    if len(orders_vec) != len(rdp_vec):  # 两个数组的长度需要相等
        raise ValueError("Input lists must have the same length.")

    # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
    # #[Mironov, 2017, Propisition 3]是RDP转DP原始的公式，可能不够紧凑
    #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

    # [Hypothesis Testing Interpretations and Rényi Differential Privacy,2020, Theorem 21 ]给了给紧凑的RDP转DP的公式如下
    # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
    # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
    eps_vec = []
    for (a, r) in zip(orders_vec, rdp_vec):
        if a < 1:
            raise ValueError("Renyi divergence order must be >=1.")
        if r < 0:
            raise ValueError("Renyi divergence must be >=0.")

        if delta ** 2 + math.expm1(-r) >= 0:  # delta的约束条件
            # In this case, we can simply bound via KL divergence:
            # delta <= sqrt(1-exp(-KL)).
            eps = 0  # No need to try further computation if we have eps = 0.
        elif a > 1.01:
            # This bound is not numerically stable as alpha->1.Thus we have a min value of alpha.
            eps = (r - (np.log(delta) + np.log(a)) / (a - 1) + np.log((a - 1) / a))
        else:
            # In this case we can't do anything. E.g., asking for delta = 0.
            eps = np.inf  # 无穷大
        eps_vec.append(eps)

    idx_opt = np.argmin(eps_vec)  # 找一个最小的
    return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


# 这里计算RDP，是没有敏感度这个参数的，本质上是有敏感度参数，分子和分母会恰好把敏感度消掉，也就是这里的参数sigma是不包括敏感度的
# 具体可以看这个的讨论：https://discuss.pytorch.org/t/how-to-adjusting-the-noise-increase-parameter-for-each-round/143548/17
def _compute_rdp(q, sigma, alpha):
    """Compute RDP of the Sampled Gaussian mechanism at order alpha.
    Args:
      q: The sampling rate.
      sigma: The std of the additive Gaussian noise.
      alpha: The order at which RDP is computed.
    Returns:
      RDP at alpha, can be np.inf.

      q==1时的公式可参考：[renyi differential privacy,2017,Proposition 7]
      0<q<1时，有以下两个公式：
      可以参考[Renyi Differential Privacy of the Sampled Gaussian Mechanism ,2019,3.3]，这篇文章中包括alpha为浮点数的计算
      公式2更为简洁的表达在[User-Level Privacy-Preserving Federated Learning: Analysis and Performance Optimization,2021,3.2和3.3]
    """
    if q == 0:
        return 0

    # no privacy
    if sigma == 0:
        return np.inf

    # q=1时相当于没有抽样，这里大家会质疑为什么没有处以（alpha-1）,其实该系数已经被消掉了(see proposition 7 of https://arxiv.org/pdf/1702.07476.pdf 1)
    if q == 1.:  # 相当于没有抽样
        return alpha / (
                2 * sigma ** 2)  # 没有抽样下参照RDP两个高斯的瑞丽分布，应该是 alpha*s  / (2 * sigma**2),这边为什么少了一个敏感度S，是默认函数敏感度为1吗，答案是敏感度和抵消了，这边的sigma里面没有敏感度的

    if np.isinf(alpha):
        return np.inf

    if float(alpha).is_integer():  # 整型
        return _compute_log_a_for_int_alpha(q, sigma, int(alpha))
    else:  # 浮点型
        return _compute_log_a_for_frac_alpha(q, sigma, alpha)


# 整型
def _compute_log_a_for_int_alpha(q, sigma, alpha):
    assert isinstance(alpha, int)
    rdp = -np.inf

    for i in range(alpha + 1):
        log_b = (
                math.log(special.binom(alpha, i))
                + i * math.log(q)
                + (alpha - i) * math.log(1 - q)
                + (i * i - i) / (2 * (sigma ** 2))
        )

        # rdp=math.exp(log_b)+math.exp(rdp)           # 当加到后面，math.exp计算的数字以小数表示，超过110000位数。超出了Double的范围，会导致溢出。所以我们用下面的方法

        # 这边其实和上面我注释的等价，这里做了一些数值超出范围的处理
        a, b = min(rdp, log_b), max(rdp, log_b)
        if a == -np.inf:  # adding 0
            rdp = b
        else:
            rdp = math.log(math.exp(
                a - b) + 1) + b  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b),这里为什么不直接exp(a) + exp(b) ，可能是容易超出数值？

    rdp = float(rdp) / (alpha - 1)
    return rdp


def _compute_log_a_for_frac_alpha(q: float, sigma: float, alpha: float) -> float:
    r"""Computes :math:`log(A_\alpha)` for fractional ``alpha``.

    Notes:
        Note that
        :math:`A_\alpha` is real valued function of ``alpha`` and ``q``,
        and that 0 < ``q`` < 1.

        Refer to Section 3.3 of https://arxiv.org/pdf/1908.10530.pdf for details.

    Args:
        q: Sampling rate of SGM.
        sigma: The standard deviation of the additive Gaussian noise.
        alpha: The order at which RDP is computed.

    Returns:
        :math:`log(A_\alpha)` as defined in Section 3.3 of
        https://arxiv.org/pdf/1908.10530.pdf.
    """
    # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
    # initialized to 0 in the log space:
    log_a0, log_a1 = -np.inf, -np.inf
    i = 0

    z0 = sigma ** 2 * math.log(1 / q - 1) + 0.5

    while True:  # do ... until loop
        coef = special.binom(alpha, i)
        log_coef = math.log(abs(coef))
        j = alpha - i

        log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
        log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

        log_e0 = math.log(0.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
        log_e1 = math.log(0.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

        log_s0 = log_t0 + (i * i - i) / (2 * (sigma ** 2)) + log_e0
        log_s1 = log_t1 + (j * j - j) / (2 * (sigma ** 2)) + log_e1

        if coef > 0:
            log_a0 = _log_add(log_a0, log_s0)
            log_a1 = _log_add(log_a1, log_s1)
        else:
            log_a0 = _log_sub(log_a0, log_s0)
            log_a1 = _log_sub(log_a1, log_s1)

        i += 1
        if max(log_s0, log_s1) < -30:
            break

    return _log_add(log_a0, log_a1) / (alpha - 1)


def _log_add(logx: float, logy: float) -> float:
    r"""Adds two numbers in the log space.

    Args:
        logx: First term in log space.
        logy: Second term in log space.

    Returns:
        Sum of numbers in log space.
    """
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx: float, logy: float) -> float:
    r"""Subtracts two numbers in the log space.

    Args:
        logx: First term in log space. Expected to be greater than the second term.
        logy: First term in log space. Expected to be less than the first term.

    Returns:
        Difference of numbers in log space.

    Raises:
        ValueError
            If the result is negative.
    """
    if logx < logy:
        raise ValueError("The result of subtraction must be non-negative.")
    if logy == -np.inf:  # subtracting 0
        return logx
    if logx == logy:
        return -np.inf  # 0 is represented as -np.inf in the log space.

    try:
        # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
        return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
    except OverflowError:
        return logx


def _log_erfc(x: float) -> float:
    r"""Computes :math:`log(erfc(x))` with high accuracy for large ``x``.

    Helper function used in computation of :math:`log(A_\alpha)`
    for a fractional alpha.

    Args:
        x: The input to the function

    Returns:
        :math:`log(erfc(x))`
    """
    return math.log(2) + special.log_ndtr(-x * 2 ** 0.5)