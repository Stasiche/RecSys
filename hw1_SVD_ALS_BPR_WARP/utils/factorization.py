from sklearn.metrics.pairwise import cosine_similarity
from utils.SGD import SGD
from scipy.special import expit

import numpy as np


class BaseFactor:
    def __init__(self, K, max_iters, lam, lr=3e-4, verbose=0):
        self.K = K
        self.max_iters = max_iters
        self.lam = lam
        self.lr = lr
        self.verbose = verbose

        self.R, self.U, self.I = None, None, None

    def _init_matrices(self, R):
        features_number, (users_number, items_number) = self.K, R.shape

        self.R = R
        self.U = np.random.uniform(0, 1/np.sqrt(features_number), size=(users_number, features_number))
        self.I = np.random.uniform(0, 1/np.sqrt(features_number), size=(features_number, items_number))

        self.U_bias = np.zeros(users_number)
        self.I_bias = np.zeros(items_number)

    def _calc_rmse(self):
        preds = (self.U @ self.I)[self.R.nonzero()]
        gt = self.R.data
        return np.sqrt(np.mean((preds - gt) ** 2))

    def _log_rmse(self, epoch):
        if not epoch % self.verbose:
            score = self._calc_rmse()
            print(f'Iter: {epoch}, score: {score}')

    def _log_auc(self, R, epoch):
        if not epoch % self.verbose:
            R = R.toarray()
            res = 0

            for user, row in enumerate(R):
                if np.count_nonzero(row != 0) != 0:
                    pos_msk = row > 0
                    neg_msk = ~row

                    preds = self.I.T @ self.U[user]
                    comp_matrix = preds[pos_msk].reshape(-1, 1) > preds[neg_msk]
                    u_auc = np.count_nonzero(comp_matrix) / (comp_matrix.shape[0] * comp_matrix.shape[1])
                    res += u_auc

            print(f'Iter: {epoch}, score: {res/R.shape[0]}')

    def get_similar_items_indxs(self, tar_item_id, n):
        sims = cosine_similarity(self.I.T)[tar_item_id]
        return np.argsort(sims)[-n:][::-1]

    def predict_for_user(self, user_id):
        return self.I.T @ self.U[user_id].T

    def recommend(self, user_id, n):
        pred = self.predict_for_user(user_id)
        return np.argsort(pred)[-n:]


class SVDLoops(BaseFactor):
    def fit(self, R):
        self._init_matrices(R)
        data_bias = np.mean(R)
        samples = [(i, j, val) for i, j, val in zip(*R.nonzero(), R.data)]

        SGD(self.U, self.I, self.R,
            self.U_bias, self.I_bias, data_bias,
            samples,
            self.lr, self.lam, self.max_iters, self.verbose)

        return self


class SVD(BaseFactor):
    def _calc_error(self, mask):
        return mask * np.asarray(self.U @ self.I - self.R)

    def fit(self, R):
        self._init_matrices(R)
        mask = (R != 0).astype(int).toarray()

        for i in range(1, self.max_iters+1):
            error = self._calc_error(mask)

            self.U -= self.lr * (error @ self.I.T + self.lam * self.U)
            self.I -= self.lr * (self.U.T @ error + self.lam * self.I)

            self._log_rmse(i)

        return self


class ALS(BaseFactor):
    def __update_user(self):
        E = self.lr * np.eye(self.K)
        II = self.I @ self.I.T
        for i in range(self.U.shape[0]):
            self.U[i] = (np.linalg.inv(II + E) @ self.I @ self.R[i].T).flatten()

    def __update_item(self):
        E = self.lr * np.eye(self.K)
        UU = self.U.T @ self.U
        for j in range(self.I.shape[1]):
            self.I[:, j] = (np.linalg.inv(UU + E) @ self.U.T @ self.R[:, j]).flatten()

    def fit(self, R):
        self._init_matrices(R)

        for epoch in range(1, self.max_iters+1):
            self.__update_user()
            self.__update_item()

            self._log_rmse(epoch)

        return self


def init(n_users, n_items, batch_size, indices, indptr):
    sampled_pos_items = np.zeros(batch_size, dtype=int)
    sampled_neg_items = np.zeros(batch_size, dtype=int)
    sampled_users = np.random.choice(n_users, size=batch_size, replace=False)

    for idx, user in enumerate(sampled_users):
        if indptr[user] != indptr[user + 1]:
            pos_items = indices[indptr[user]: indptr[user + 1]]

            pos_item = np.random.choice(pos_items)
            neg_item = np.random.choice(n_items)
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item

    return sampled_users, sampled_pos_items, sampled_neg_items


class BPR(BaseFactor):
    def __update(self, u, i, j):
        user_u = self.U[u]
        item_i = self.I[:, i].T
        item_j = self.I[:, j].T

        r_uij = np.sum(user_u * (item_i - item_j), axis=1)

        sigmoid = np.tile(expit(-r_uij), (self.K, 1)).T
        sigmoid_u = sigmoid * user_u

        grad_u = sigmoid * (item_j - item_i) + self.lam * user_u
        grad_i = -sigmoid_u + self.lam * item_i
        grad_j = sigmoid_u + self.lam * item_j
        self.U[u] -= self.lr * grad_u
        self.I[:, i] -= self.lr * grad_i.T
        self.I[:, j] -= self.lr * grad_j.T

    def fit(self, R, batch_size):
        self._init_matrices(R)

        indptr = R.indptr
        indices = R.indices

        for epoch in range(1, self.max_iters+1):
            users, items_pos, items_neg = init(R.shape[0], R.shape[1], batch_size, indices, indptr)
            self.__update(users, items_pos, items_neg)

            self._log_auc(R, epoch)
        return self
