from typing import NamedTuple, ClassVar
import numpy
import numpy as np

from ..core import Graph, Node, Ref, Event, Loop, Defn, define, steps

#  ------------------

def window_mask(v, t, at, offset: float | None = None):
    if offset is not None:
        return v[(t >= at - offset)&(t < at )]
    return v[t > at]

def window_null_mask(v, t, at):
    return v[(t > at) & ~np.isnan(v)]

#  ------------------


class sum_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: float | None



class sum(sum_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
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
        cls, v: Ref.Scalar_F64, window: float | None = None
    ):
        return cls(cls.DEF.name, v=v, window=window)

    def __call__(
        self, event: Event, graph: Graph
    ):
        window = event.t + 1 if self.window is None else self.window
        v = graph.select(self.v, event, where=dict(
            t=lambda t: t >= event.t - window
        ))
        return np.NAN if not len(v) or np.isnan(v[-1]) else numpy.nansum(v)

#  ------------------


class mean_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: float | None
    offset: float | None
    transform: str | None


class mean(mean_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
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
        window: float | None = None,
        offset: float | None = None,
        transform: str | None = None,
    ):
        return cls(cls.DEF.name, v=v, window=window, offset=offset, transform=transform)

    def __call__(
        self, event: Event, graph: Graph
    ):
        window = event.t + 1 if self.window is None else self.window
        v = graph.select(self.v, event, where=dict(
            t=lambda t: t >= event.t - window
        ))
        return np.NAN if not len(v) or np.isnan(v[-1]) else numpy.nanmean(v)


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
        pass


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
        pass


#  ------------------


class std_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: float | None


class std(std_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
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
        cls, v: Ref.Scalar_F64, window: float | None = None
    ):
        return cls(cls.DEF.name, v=v, window=window)

    def __call__(
        self, event: Event, graph: Graph
    ):
        window = event.t + 1 if self.window is None else self.window
        v = graph.select(self.v, event, where=dict(
            t=lambda t: t >= event.t - window
        ))
        return np.NAN if not len(v) or np.isnan(v[-1]) else numpy.nanstd(v)



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
        pass


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
        pass


#  ------------------



class rms_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    window: float | None


class rms(rms_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
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
        cls, v: Ref.Scalar_F64, window: float | None = None
    ):
        return cls(cls.DEF.name, v=v, window=window)

    def __call__(
        self, event: Event, graph: Graph
    ):
        window = event.t + 1 if self.window is None else self.window
        v = graph.select(self.v, event, where=dict(
            t=lambda t: t >= event.t - window
        ))
        return (
            np.NAN 
            if not len(v) or np.isnan(v[-1]) 
            else np.sqrt(np.nanmean(np.square(v)))
        )

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
        pass


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
        pass


#  ------------------

# hinge (as per turn, av delta around point), incl w, ew

# variance

# quantile

# iqr

# max, min incl w, ew

# skew, kurtosis

#  ------------------


class ex_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64


class ex(ex_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, ex_kw
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
        pass


#  ------------------

# corr, cov, levy, beta (1D)

#  ------------------


class cov_kw(NamedTuple):
    type: str
    #
    v1: Ref.Scalar_F64
    v2: Ref.Scalar_F64
    window: float | None
    mu_1: Ref.Scalar_F64 | None
    mu_2: Ref.Scalar_F64 | None
    offset_1: float | None
    offset_2: float | None
    shuffle: bool
    bootstrap: int | None


class cov(cov_kw, Node.Scalar_F64):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
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
        window: float | None = None,
        mu_1: Ref.Scalar_F64 | None = None,
        mu_2: Ref.Scalar_F64 | None = None,
        offset_1: float | None=None,
        offset_2: float | None=None,
        shuffle: bool = False,
        bootstrap: int | None = None,
    ):
        return cls(
            cls.DEF.name,
            v1=v1,
            v2=v2,
            window=window,
            mu_1=mu_1,
            mu_2=mu_2,
            offset_1=offset_1,
            offset_2=offset_2,
            shuffle=shuffle,
            bootstrap=bootstrap,
        )

    def __call__(
        self, event: Event, graph: Graph
    ):
        window = event.t + 1 if self.window is None else self.window
        # TODO: assert aligned?
        t1, v1 = self.v1.history(graph).last_before(event.t)
        t2, v2 = self.v2.history(graph).last_before(event.t)
        
        if np.isnan(v1[-1]):
            return np.NAN
            
        if np.isnan(v2[-1]):
            return np.NAN

        v1 = window_mask(v1, t1, event.t - window, offset=self.offset_1)
        v2 = window_mask(v2, t2, event.t - window, offset=self.offset_2)

        if not len(v1) or not len(v2):
            return np.NAN

        # if len(v1) != len(v2):
        #     return np.NAN

        if self.shuffle:
            np.random.shuffle(v1)

        if not self.bootstrap:
            n_runs = 1
            n_vs = len(v1)
        else:
            n_runs = self.bootstrap
            n_vs = int(len(v1) / n_runs)
        
        res = 0
        for i in range(n_runs):
            
            vv1 = v1[i*n_vs:(i+1) * n_vs]
            vv2 = v2[i*n_vs:(i+1) * n_vs]

            if self.mu_1 is None:
                mu1 = np.NAN if not len(vv1) else numpy.nanmean(vv1)
            elif isinstance(self.mu_1, (int, float)):
                mu1 = self.mu_1
            else:
                mu1 = graph.select(self.mu_1, event)[-1]
            if self.mu_2 is None:
                mu2 = np.NAN if not len(vv2) else numpy.nanmean(vv2)
            elif isinstance(self.mu_2, (int, float)):
                mu2 = self.mu_2
            else:
                mu2 = graph.select(self.mu_2, event)[-1]
            if np.isnan(mu1) and np.isnan(mu2):
                return np.NAN
            res += (numpy.nanmean(
                (vv1 - mu1) * (vv2 - mu2)
                # assume elementwise?
            )/ n_runs)
        return res

#  ------------------


class pca_kw(NamedTuple):
    type: str
    #
    vs: tuple[Ref.Scalar_F64, ...]
    window: float | None
    keep: int | None
    mus: tuple[Ref.Scalar_F64, ...] | None
    signs: tuple[int | None] | None
    centre: bool

    # TODO: window is constant, but take offsets
    # so for forward vs backward
    # can take [window][offset] vs [window]

    # eg. via the covar cell, and do pca on the covar matrix
    # then to realign for the graph
    # you need a forward shift on the date index
    # but that in theory is just date + n days (optionally over a given calendar)

class pca(pca_kw, Node.Scalar_F641D):
    """
    scalar mean (optional rolling window)
    v: Ref.Scalar_F64
    window: float | None
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
        Node.Scalar_F641D, pca_kw
    )

    @classmethod
    def new(
        cls,
        vs: tuple[Ref.Scalar_F64, ...],
        window: float | None=None,
        keep: int | None=None,
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
        window = event.t + 1 if self.window is None else self.window

        vs = graph.select(
            self.vs,
            event,
            t=False,
            where=dict(t = lambda t: t >= event.t - window),
            null=False
        )

        # TODO: assert aligned?

        mus = [None for _ in vs] if self.mus is None else self.mus
        assert len(mus) == len(vs), dict(mus=mus, vs=vs)
        mus = list(map(
            lambda v_mu: (lambda v, mu: (
                graph.select(mu, event, i = -1) if mu is not None
                else np.NAN if not len(v)
                else numpy.nanmean(v)
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
                        self=self,
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
    v: Ref.Scalar_F64
    factor: int

class pca_scale(pca_scale_kw, Node.Scalar_F64):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F64, pca_scale_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
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
        keep = n.keep
        vs = graph.select(self.v, event, t=False, i = -1, null = False)
        vs = vs.reshape(
            int(vs.shape[0] / keep), keep, 
        )
        return vs[0,self.factor]

class pca_weights_kw(NamedTuple):
    type: str
    #
    v: Ref.Scalar_F64
    factor: int

class pca_weights(pca_weights_kw, Node.Scalar_F641D):

    DEF: ClassVar[Defn] = define(
        Node.Scalar_F641D, pca_weights_kw
    )

    @classmethod
    def new(
        cls,
        v: Ref.Scalar_F64,
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
        keep = n.keep
        vs = graph.select(self.v, event, t=False, i=-1, null=False)
        vs = vs.reshape(
            int(vs.shape[0] / keep), keep, 
        )
        w = vs[1:,self.factor]
        return w

#  ------------------