import numpy as np


def safe_softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    ex_fenzi = np.exp(x - x_max)

    ex_fenmu = np.sum(ex_fenzi, axis=-1, keepdims=True)

    return ex_fenzi / ex_fenmu


def log_softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    ex_x = np.exp(x - x_max)
    '''
    log(Σⱼ exp(xⱼ))
    = log(Σⱼ exp(xⱼ - m + m))
    = log(Σⱼ exp(xⱼ - m) · exp(m))
    = log(exp(m) · Σⱼ exp(xⱼ - m))
    = m + log(Σⱼ exp(xⱼ - m))  
    '''
    log_x = x_max + np.log(np.sum(ex_x, axis=-1, keepdims=True))

    return x - log_x


if __name__ == "__main__":
    x = np.array([1001, 1000, 999])
    print(safe_softmax(x))
    print(log_softmax(x))
