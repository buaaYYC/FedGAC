
import numpy as np



def __getDirichlet__(seed, alpha,):

    np.random.seed(seed)
    for i in range(10):
        proportions = np.random.dirichlet(np.repeat(alpha, 10))
        print(proportions)

    
    return proportions


# 根据狄利克雷分布分配数据
train_prop= __getDirichlet__(1,0.1)
train_prop= __getDirichlet__(1,0.1)

