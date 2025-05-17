import numpy as np
from numpy.testing import suppress_warnings

from .common import Benchmark, XPBenchmark, safe_import

with safe_import():
    from scipy.cluster.hierarchy import linkage, is_isomorphic
    from scipy.cluster.vq import kmeans, kmeans2, vq


class HierarchyLinkage(Benchmark):
    params = ['single', 'complete', 'average', 'weighted', 'centroid',
              'median', 'ward']
    param_names = ['method']

    def __init__(self):
        rnd = np.random.RandomState(0)
        self.X = rnd.randn(2000, 2)

    def time_linkage(self, method):
        linkage(self.X, method=method)


class IsIsomorphic(XPBenchmark):
    nobs = [100, 1_000, 10_000, 100_000, 1_000_000]
    param_names = ["backend", "device", "nobs"]
    params = [
        (backend, device, n)
        for n in nobs 
        for backend, device in XPBenchmark.backends_devices
    ]

    def __init__(self):
        super().__init__()

        # Initialize 5 random isomorphic clusters with the
        # given number of observations (5 * nobs points)
        self.args = {}
        nclusters = 5
        for nobs in self.nobs:
            a = (np.random.rand(nobs) * nclusters).astype(int)
            b = np.zeros_like(a)
            P = np.random.permutation(nclusters)
            for i in range(0, a.shape[0]):
                b[i] = P[a[i]]
            self.args[nobs] = a, b

    def setup(self, backend, device, nobs):
        super().setup(backend, device)

        a, b = self.args[nobs]
        self.a = self.xp.asarray(a)
        self.b = self.xp.asarray(b)

        if backend == "dask.array":
            self.f = lambda a, b: is_isomorphic(a, b).compute()
        elif backend == "jax.numpy":
            import jax
            j = jax.jit(is_isomorphic)
            self.f = lambda a, b: j(a, b).block_until_ready()
            self.f(self.a, self.b)  # warm up JIT
        else:
            self.f = is_isomorphic

    def time_isomorphic(self, backend, device, nobs):
        self.f(self.a, self.b)

    
class KMeans(Benchmark):
    params = [2, 10, 50]
    param_names = ['k']

    def __init__(self):
        rnd = np.random.RandomState(0)
        self.obs = rnd.rand(1000, 5)

    def time_kmeans(self, k):
        kmeans(self.obs, k, iter=10)


class KMeans2(Benchmark):
    params = [[2, 10, 50], ['random', 'points', '++']]
    param_names = ['k', 'init']

    def __init__(self):
        rnd = np.random.RandomState(0)
        self.obs = rnd.rand(1000, 5)

    def time_kmeans2(self, k, init):
        with suppress_warnings() as sup:
            sup.filter(UserWarning,
                       "One of the clusters is empty. Re-run kmeans with a "
                       "different initialization")
            kmeans2(self.obs, k, minit=init, iter=10)


class VQ(Benchmark):
    params = [[2, 10, 50], ['float32', 'float64']]
    param_names = ['k', 'dtype']

    def __init__(self):
        rnd = np.random.RandomState(0)
        self.data = rnd.rand(5000, 5)
        self.cbook_source = rnd.rand(50, 5)

    def setup(self, k, dtype):
        self.obs = self.data.astype(dtype)
        self.cbook = self.cbook_source[:k].astype(dtype)

    def time_vq(self, k, dtype):
        vq(self.obs, self.cbook)
