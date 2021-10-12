import numpy as np


def error_function(user_reprs, item_reprs, bias, gt_value):
    return user_reprs @ item_reprs + bias - gt_value


def rmse(U, I, R):
    preds = (U @ I)[R.nonzero()]
    gt = R.data
    return np.sqrt(np.mean((preds - gt)**2))


def SGD(U, I, R, U_bias, I_bias, data_bias, samples, lr, lam, max_iterations, verbose):
    for epoch in range(1, max_iterations+1):
        for ind in np.random.choice(len(samples), size=len(samples)):
            u, i, v = samples[ind]
            error = error_function(U[u], I[:, i], U_bias[u] + I_bias[i] + data_bias, v)

            U[u] -= lr * (error*I[:, i] + lam*U[u])
            I[:, i] -= lr * (error*U[u] + lam*I[:, i])

            U_bias[u] -= lr * (error + lam*U_bias[u])
            I_bias[i] -= lr * (error + lam*I_bias[i])

        if not (epoch % verbose):
            score = rmse(U, I, R)
            print(f'Iter: {epoch}, score: {score}')
