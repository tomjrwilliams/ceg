
import numpy as np

def svd_pca(
    vs: list[np.ndarray] | np.ndarray, # [n_obs, n_feats]
    mus: list[float] | np.ndarray | None = None,
    keep: int | None = None,
    centre: bool = False,
    signs: None | tuple[int | None, ...] = None,
):

    if isinstance(vs, list):
        vs = np.vstack([np.expand_dims(v, 0) for v in vs]).T

    n_obs = vs.shape[0]
    n_features = vs.shape[1]

    if keep is None:
        keep = n_features
    assert isinstance(keep, int), keep

    if centre:
        if mus is None:
            mus = np.nanmean(vs, axis = 0)
        elif isinstance(mus, list):
            mus = np.array(mus)
        
        assert isinstance(mus, np.ndarray), type(mus)
        assert mus.shape[0] == vs.shape[1], dict(
            mus=mus.shape, vs=vs.shape
        )
        vs = vs - np.expand_dims(mus, 0)

    vs = vs[~np.any(np.isnan(vs), axis=1)].T

    if vs.shape[1] < vs.shape[0]:
        # less observations than features
        e = np.array(
            [np.nan for _ in range(keep)]
        )
        u = np.array([np.nan for _ in range(n_features)])
        U = np.hstack(
            [
                np.expand_dims(u, 1)
                for _ in range(keep)
            ]
        )
        return e, U

    # vs = (features, observations)
    U, e, _ = np.linalg.svd(vs, full_matrices=False)

    if signs is not None:
        for i, s in enumerate(signs):
            if s is None:
                continue
            elif s < 0:
                if U[-1, i] < 0:
                    continue
            elif s > 0:
                if U[-1, i] > 0:
                    continue
            U[:, i] *= -1

            # # TODO: numba?

            # if s is None:
            #     continue
            # elif s == 0:
            #     raise ValueError(
            #         dict(
            #             message="signs start from 1 for symmetry",
            #             self=self,  # type: ignore (wants dicts to be the same type)
            #         )
            #     )
            # elif s < 0:
            #     s = -1 * (s + 1)
            #     if U[s, i] < 0:
            #         continue
            # else:
            #     s -= 1
            #     if U[s, i] > 0:
            #         continue
            # U[:, s] *= -1

    if keep is not None:
        U = U[:, : keep]
        e = e[: keep]
        # Vt = Vt[:, :keep]
    
    # PCs would be the columns of: U*S^(1/2)
    
    # cols of U = unit PCs
    # S = np.diag(s)
    # Mhat = np.dot(U, np.dot(S, V.T))

    return e, U
    
# singular value decomposition factorises your data matrix such that:
#
#   M = U*S*V.T     (where '*' is matrix multiplication)
#
# * U and V are the singular matrices, containing orthogonal vectors of
#   unit length in their rows and columns respectively.
#
# * S is a diagonal matrix containing the singular values of M - these
#   values squared divided by the number of observations will give the
#   variance explained by each PC.
#
# * if M is considered to be an (observations, features) matrix, the PCs
#   themselves would correspond to the rows of S^(1/2)*V.T. if M is
#   (features, observations) then the PCs would be the columns of
#   U*S^(1/2).
#
# * since U and V both contain orthonormal vectors, U*V.T is equivalent
#   to a whitened version of M.

#  ------------------
