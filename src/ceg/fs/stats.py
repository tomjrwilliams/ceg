from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from ..core import Graph, Node, Ref, Event, Loop, Defn, define, steps

#  ------------------


class sum_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: int



class sum(sum_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: int
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, sum_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64, window: int
    ):
        return cls(cls.DEF.name, v=v, window=window)

    def __call__(
        self, event: Event, graph: Graph
    ):
        return np.nansum(
            self.v.history(graph).last_n_before(self.window, event.t)
        )

#  ------------------


class mean_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: int


class mean(mean_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: int
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, mean_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64,
        window: int,
    ):
        return cls(cls.DEF.name, v=v, window=window)

    def __call__(
        self, event: Event, graph: Graph
    ):
        return np.nanmean(
            self.v.history(graph).last_n_before(self.window, event.t)
        )


#  ------------------


class mean_w_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class mean_w(mean_w_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, mean_w_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            cls.DEF.name,
            v=v,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        raise ValueError(self)


#  ------------------


class mean_ew_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class mean_ew(mean_ew_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, mean_ew_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            cls.DEF.name,
            v=v,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        raise ValueError(self)


#  ------------------


class std_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: int


class std(std_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: int
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(std.new(r))
    ...     mu_3 = bind(std.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, std_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64, window: int
    ):
        return cls(cls.DEF.name, v=v, window=window)

    def __call__(
        self, event: Event, graph: Graph
    ):
        return np.nanstd(
            self.v.history(graph).last_n_before(self.window, event.t)
        )



#  ------------------


class std_w_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class std_w(std_w_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, std_w_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            cls.DEF.name,
            v=v,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        raise ValueError(self)


#  ------------------


class std_ew_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class std_ew(std_ew_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, std_ew_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            cls.DEF.name,
            v=v,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        raise ValueError(self)


#  ------------------



class rms_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: int


class rms(rms_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: int
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(rms.new(r))
    ...     mu_3 = bind(rms.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, rms_kw
    )

    @classmethod
    def new(
        cls, v: Ref.Scalar_F64, window: int
    ):
        return cls(cls.DEF.name, v=v, window=window)

    def __call__(
        self, event: Event, graph: Graph
    ):
        
        return np.sqrt(np.nanmean(np.square(
            self.v.history(graph).last_n_before(self.window, event.t)
        )))

#  ------------------


class rms_w_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class rms_w(rms_w_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, rms_w_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            cls.DEF.name,
            v=v,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        raise ValueError(self)


#  ------------------


class rms_ew_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class rms_ew(rms_ew_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, rms_ew_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
    ):
        return cls(
            cls.DEF.name,
            v=v,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        raise ValueError(self)


#  ------------------

# hinge (as per turn, av delta around point), incl w, ew

# variance

# quantile

# iqr

# max, min incl w, ew

# skew, kurtosis

#  ------------------

# corr, cov, levy, beta (1D)

#  ------------------


class cov_kw(NamedTuple):
    type: str
    #
    v1: Ref.Scalar_F64
    v2: Ref.Scalar_F64
    window: int
    mu_1: Ref.Scalar_F64 | None
    mu_2: Ref.Scalar_F64 | None


class cov(cov_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: int
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, cov_kw
    )

    @classmethod
    def new(
        cls,
        v1: Ref.Scalar_F64,
        v2: Ref.Scalar_F64,
        window: int,
        mu_1: Ref.Scalar_F64 | None = None,
        mu_2: Ref.Scalar_F64 | None = None,
    ):
        return cls(
            cls.DEF.name,
            v1=v1,
            v2=v2,
            window=window,
            mu_1=mu_1,
            mu_2=mu_2,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        v1 = self.v1.history(graph).last_n_before(self.window, event.t)
        v2 = self.v2.history(graph).last_n_before(self.window, event.t)

        if self.mu_1 is not None:
            mu_1 = self.mu_1.history(graph).last_before(event.t)
        else:
            mu_1 = np.nanmean(v1)
        
        if self.mu_2 is not None:
            mu_2 = self.mu_2.history(graph).last_before(event.t)
        else:
            mu_2 = np.nanmean(v2)
    
        res = numpy.nanmean(
            (v1 - mu_1) * (v2 - mu_2)
        )

        return res

#  ------------------


class pca_kw(NamedTuple):
    type: str
    #
    vs: tuple[Ref.Scalar_F64, ...]
    window: int
    keep: int
    mus: tuple[Ref.Scalar_F64, ...] | None
    signs: tuple[int | None] | None
    centre: bool

class pca(pca_kw, Node.Vector_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: int
    >>> g = Graph.new()
    >>> from . import rand
    >>> _ = rand.rng(seed=0, reset=True)
    >>> g, r = gaussian.bind(g)
    >>> with g.implicit() as (bind, done):
    ...     mu = bind(mean.new(r))
    ...     mu_3 = bind(mean.new(r, window=3))
    ...     g = done()
    ...
    >>> g, es = g.steps(Event(0, r), n=18)
    >>> list(numpy.round(g.select(r, es[-1]), 2))
    [0.13, -0.01, 0.63, 0.74, 0.2, 0.56]
    >>> list(numpy.round(g.select(mu, es[-1]), 2))
    [0.13, 0.06, 0.25, 0.37, 0.34, 0.38]
    >>> list(
    ...     numpy.round(g.select(mu_3, es[-1]), 2)
    ... )
    [0.13, 0.06, 0.25, 0.46, 0.53, 0.5]
    """

    DEF: ClassVar[Defn] = define(
        Node.Vector_F64, pca_kw
    )

    @classmethod
    def new(
        cls,
        vs: tuple[Ref.Scalar_F64, ...],
        window: int,
        keep: int,
        mus: tuple[Ref.Scalar_F64, ...] | None=None,
        signs: tuple[int | None] | None = None,
        centre: bool = False,
    ):
        return cls(
            cls.DEF.name, vs=vs, window=window, keep=keep, mus=mus,signs=signs, centre=centre
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        vs = tuple(map(
            lambda v: v.history(graph).last_n_before(
                self.window, event.t
            ),
            self.vs
        ))

        # TODO: assert aligned?

        mus = [None for _ in vs] if self.mus is None else self.mus
        assert len(mus) == len(vs), dict(mus=mus, vs=vs)
        mus = list(map(
            lambda v_mu: (lambda v, mu: (
                numpy.nanmean(v)
                if mu is None
                else np.NAN if not len(v)
                else mu.history(graph).last_before(event.t)
            ))(*v_mu),
            zip(vs, mus)
        ))

        if self.centre:
            vs = [v - mu for v, mu in zip(vs, mus)]

        vs = numpy.vstack([
            np.expand_dims(v, 0) for v in vs
        ]).T
        
        vs = vs[
            ~np.any(np.isnan(vs), axis=1)
        ].T

        if vs.size <= len(mus):
            e = np.array([np.NAN for _ in range(self.keep)])
            u = np.array([np.NAN for _ in mus])
            U = np.hstack([
                np.expand_dims(u, 1)
                for _ in range(self.keep)
            ])
            return np.vstack([
                np.expand_dims(e, 0),
                U
            ]).reshape(
                (self.keep * (len(mus) + 1))
            )

        # (window, n variables)

        U, e, _ = np.linalg.svd(vs, full_matrices=False)

        if self.signs is not None:
            for i, s in enumerate(self.signs):
                if s is None:
                    continue
                elif s == 0:
                    raise ValueError(dict(
                        message="signs start from 1 for symmetry",
                        self=self, # type: ignore (wants dicts to be the same type)
                    ))
                elif s < 0:
                    s = -1 * (s + 1)
                    if U[s, i] < 0:
                        continue
                else:
                    s -= 1
                    if U[s, i] > 0:
                        continue
                U[:, s] *= -1

        if self.keep is not None:
            U = U[:, :self.keep]
            e = e[:self.keep]
            # Vt = Vt[:, :self.keep]

        # TODO: now we have tuple returns, use them
        # rather than flattening to a single vec

        # even a matrix would be clearer

        try:
            return np.vstack([
                np.expand_dims(e, 0),
                U
            ]).reshape(
                (self.keep * (len(mus) + 1))
            )
        except:
            raise ValueError(vs, U, e)

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

# S = np.diag(s)
# Mhat = np.dot(U, np.dot(S, V.T))

class pca_scale_kw(NamedTuple):
    type: str
    #
    v: Ref.Vector_F64
    factor: int

class pca_scale(pca_scale_kw, Node.Scalar_F64):

    # TODO: change pca to tuple, then this isn't even required?
    # though perhaps worth keeping - this still is useful for unpacking a specific element of the eigenvector?

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, pca_scale_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Vector_F64,
        factor: int,
    ):
        return cls(
            cls.DEF.name, v=v, factor=factor
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        # first row is the eignenvalues
        n = graph.nodes[self.v.i]
        assert isinstance(n, pca), n
        keep = n.keep
        vs = self.v.history(graph).last_before(event.t)
        vs = vs.reshape(
            int(vs.shape[0] / keep), keep, 
        )
        return vs[0,self.factor]

class pca_weights_kw(NamedTuple):
    type: str
    #
    v: Ref.Vector_F64
    factor: int

class pca_weights(pca_weights_kw, Node.Vector_F64):

    DEF: ClassVar[Defn] = define(
        Node.Vector_F64, pca_weights_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Vector_F64,
        factor: int,
    ):
        return cls(
            cls.DEF.name, v=v, factor=factor
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        # first row is eigenvalues. then weights are cols of remainder
        n = graph.nodes[self.v.i]
        assert isinstance(n, pca), n
        keep = n.keep
        vs = self.v.history(graph).last_before(event.t)
        vs = vs.reshape(
            int(vs.shape[0] / keep), keep, 
        )
        w = vs[1:,self.factor]
        return w

#  ------------------