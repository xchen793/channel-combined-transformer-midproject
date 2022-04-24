import numpy as np

def awgn(x, snr, seed=7):
    '''
    Additive White Gaussian Noise
    :param x: original signal
    :param snr: SNR
    :return: signal with additive white gaussian noise
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise
