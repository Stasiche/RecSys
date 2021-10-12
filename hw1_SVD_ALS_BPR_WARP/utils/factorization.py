import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.SGD import SGD
from scipy.special import expit


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

    def _calc_error(self, mask):
        return mask*np.asarray(self.U @ self.I - self.R)

    def _log(self, epoch):
        if not epoch % self.verbose:
            score = self._calc_rmse()
            print(f'Iter: {epoch}, score: {score}')

    def get_similar_items_indxs(self, tar_item_id, n):
        sims = cosine_similarity(self.I)[tar_item_id]
        return np.argsort(sims)[-n:][::-1]

    def recommend(self, user_id, n):
        pred = self.I.T @ self.U[user_id]
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
    def fit(self, R):
        self._init_matrices(R)
        mask = (R != 0).astype(int).toarray()

        for i in range(1, self.max_iters+1):
            error = self._calc_error(mask)

            self.U -= self.lr * (error @ self.I.T + self.lam * self.U)
            self.I -= self.lr * (self.U.T @ error + self.lam * self.I)

            self._log(i)

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
        for j in range(self.I.shape[0]):
            self.I[:, j] = (np.linalg.inv(UU + E) @ self.U.T @ self.R[:, j]).flatten()

    def fit(self, R):
        self._init_matrices(R)

        for epoch in range(1, self.max_iters+1):
            self.__update_user()
            self.__update_item()

            self._log(epoch)

        return self
