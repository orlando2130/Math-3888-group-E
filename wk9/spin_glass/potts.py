import numpy as np
import networkx as nx
import scipy.sparse as sp
from numba import njit

@njit
def _seed_numba(seed):
    np.random.seed(seed)
    
@njit
def _heat_bath_sweep(indptr, indices, data, k, sigma, K, order,
                     gamma, beta, two_M, q, a, p):
    """One heat-bath sweep over `order`. Modifies sigma and K in place.

    a, p are preallocated scratch arrays of length q.
    Returns the number of nodes whose label changed.
    """
    n_changed = 0
    for l in order:
        old = sigma[l]
        K[old] -= k[l]                       # remove l from its community

        coef = gamma * k[l] / two_M
        for s in range(q):
            a[s] = -coef * K[s]              # null-model term
        for e in range(indptr[l], indptr[l + 1]):
            j = indices[e]
            if j != l:
                a[sigma[j]] += data[e]       # edge term

        amax = a[0]
        argmax = 0
        for s in range(1, q):
            if a[s] > amax:
                amax = a[s]
                argmax = s

        total = 0.0
        for s in range(q):
            p[s] = np.exp(beta * (a[s] - amax))
            total += p[s]

        u = np.random.random() * total       # inverse-CDF sampling
        new = argmax                         # fallback for rounding at the tail
        acc = 0.0
        for s in range(q):
            acc += p[s]
            if u < acc:
                new = s
                break

        sigma[l] = new
        K[new] += k[l]
        if new != old:
            n_changed += 1
    return n_changed


class PottsNumpy:
    """Potts-model community detection for large sparse graphs.

    Parameters
    ----------
    G : nx.Graph
        Network to partition. Edge weights are read from the 'weight' attribute
        (default 1). Self-loops are dropped.
    gamma : float, optional
        Resolution parameter. Larger (smaller) gamma favours smaller (larger)
        communities. By default 1, where minimising H maximises modularity.
    q : int, optional
        Maximum number of communities (the number of Potts states). The number
        actually occupied may be smaller. Per-node cost scales with q. By default 100.
    seed : int, optional
        Seeds both the NumPy generator (visit order, initial state) and the
        Numba generator (heat-bath sampling).
    init : array-like of int, shape (n,), optional
        Initial labels in [0, q). Random if None. A decent starting partition
        (e.g. from Louvain) converges much faster than random labels.
    """
    
    def __init__(self, G: nx.Graph, gamma=1.0, q=100, seed=None, init=None) -> None:
        
        self.graph = G
        self.gamma = gamma
        self.q = q
        self.nodelist = list(G.nodes())
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            _seed_numba(int(seed))
        
        # store adjacency matrix as a sparse array (should require less memory)
        A = sp.csr_array(nx.to_scipy_sparse_array(
            G, nodelist=self.nodelist, format="csr", dtype=np.float64
            ))
        A.setdiag(0) # no self links
        A.eliminate_zeros()
        A.sort_indices()
        self.A = A

        self.k = np.asarray(A.sum(axis=1)).ravel()
        self.two_M = float(self.k.sum())
        self.M = self.two_M / 2
        if self.two_M == 0:
            raise ValueError("Graph has no edges.")
        self._k2_sum = float(np.sum(self.k ** 2))      # sum_i k_i^2
        
        # initial state is random if not specified
        n = len(self.nodelist)
        if init is None:
            self.sigma = self.rng.integers(0, self.q, n).astype(np.int64)
        else:
            self.sigma = np.asarray(init, dtype=np.int64).copy()
            if self.sigma.shape != (n,) or self.sigma.min() < 0 or self.sigma.max() >= self.q:
                raise ValueError("init must have shape (n,) with labels in [0, q).")
        
        # We will use these later
        self.K = np.bincount(self.sigma, weights=self.k, minlength=self.q)
        self._a = np.empty(self.q)
        self._p = np.empty(self.q)
        self.n_changed = 0
            
    """base definition of Hamiltonian"""
    
    def _internal_weight(self):
        """sum over ORDERED pairs (i, j) in the same community of A_ij (O(M))."""
        coo = self.A.tocoo() # to coordinate format
        same = self.sigma[coo.row] == self.sigma[coo.col]
        return coo.data[same].sum()

    def hamiltonian(self):
        """Energy of the current state, in O(M + n).

        H = -1/2 * [ sum_{ij same group} A_ij - gamma * (sum_s K_s^2 - sum_i k_i^2) / 2M ]
        """
        K = np.bincount(self.sigma, weights=self.k, minlength=self.q)
        within_A = self._internal_weight()
        within_P = (np.sum(K ** 2) - self._k2_sum) / self.two_M   # i != j only
        return -0.5 * (within_A - self.gamma * within_P)

    def hamiltonian_dense(self):
        """Brute-force O(n^2) reference implementation. Use only for testing."""
        A = self.A.toarray()
        P = np.outer(self.k, self.k) / self.two_M
        J = A - self.gamma * P
        delta = self.sigma[:, None] == self.sigma[None, :]
        np.fill_diagonal(delta, False)
        return -0.5 * np.sum(J * delta)          # 1/2: each unordered pair once

    def modularity(self):
        """Newman modularity Q of the current partition (independent of gamma)."""
        K = np.bincount(self.sigma, weights=self.k, minlength=self.q)
        return (self._internal_weight() - np.sum(K ** 2) / self.two_M) / self.two_M
    
    """definition of cohesion and adhesion with corresponding computation of Hamiltonian"""
    
    def _blocks(self):
        """Community-level coupling matrix D = M_rs - gamma * E_rs.

        Returns
        -------
        labels : ndarray, shape (r,)
            Occupied community labels (rows/cols of D map back to these).
        D : ndarray, shape (r, r)
            M = S.T A S and E = Kc Kc^T / 2M, with S a sparse one-hot membership
            matrix and Kc the community degree sums. Cost O(M + r^2).
        """
        labels, c = np.unique(self.sigma, return_inverse=True)
        r = len(labels)
        n = len(c)
        S = sp.csr_array((np.ones(n), (np.arange(n), c)), shape=(n, r))
        M = (S.T @ self.A @ S).toarray()
        Kc = np.bincount(c, weights=self.k, minlength=r)
        E = np.outer(Kc, Kc) / self.two_M
        return labels, M - self.gamma * E

    def cohesion(self):
        """Cohesion c_s of each occupied community (pairs i < j inside s).

        The diagonal block of D counts every internal pair twice and includes
        the i = j terms, so we halve it and add back the P_ii self-terms.
        """
        labels, c = np.unique(self.sigma, return_inverse=True)
        _, D = self._blocks()
        k2 = np.bincount(c, weights=self.k ** 2, minlength=len(labels))
        return labels, 0.5 * (np.diag(D) + self.gamma * k2 / self.two_M)

    def adhesion(self):
        """Adhesion a_rs between occupied communities, strictly upper-triangular.

        Off-diagonal blocks already count each inter-community pair once, so no
        factor 1/2 and no diagonal.
        """
        labels, D = self._blocks()
        return labels, np.triu(D, k=1)

    def hamiltonian_cohesion(self):
        """H = -sum_s c_s."""
        _, c = self.cohesion()
        return -c.sum()

    def hamiltonian_adhesion(self):
        """H = sum_{r<s} a_rs - T, where T = sum_{i<j} J_ij over all pairs.

        T is a partition-independent constant, so this differs from the
        'adhesion' of the partition only by that offset; it equals H exactly.
        """
        _, a = self.adhesion()
        T = 0.5 * (self.two_M - self.gamma * (self.two_M - self._k2_sum / self.two_M))
        return a.sum() - T
    
    """heat bath update rules"""
    
    def heat_bath_probability(self, l, beta):
        """Heat-bath probabilities for node l (pure NumPy; for inspection/tests).

        a_l,alpha = sum_{j != l} A_lj delta(sigma_j, alpha) - gamma k_l K_alpha^(-l) / 2M
        p(alpha)  = exp(beta a_l,alpha) / sum_s exp(beta a_l,s)

        K^(-l) is the community degree sums with node l removed. Cost O(deg(l) + q).
        """
        s, e = self.A.indptr[l], self.A.indptr[l + 1]
        nbrs, w = self.A.indices[s:e], self.A.data[s:e]
        mask = nbrs != l
        A_to_group = np.bincount(self.sigma[nbrs[mask]], weights=w[mask], minlength=self.q)

        K = self.K.copy()
        K[self.sigma[l]] -= self.k[l]
        a = A_to_group - self.gamma * self.k[l] * K / self.two_M

        w_ = beta * a
        w_ -= w_.max()
        p = np.exp(w_)
        return p / p.sum()

    def heat_bath_sweep(self, beta):
        """One sweep: update every node once, in random order (Numba-compiled).

        Returns the updated state vector; the number of changed labels is
        stored in ``self.n_changed``.
        """
        order = self.rng.permutation(len(self.sigma))
        self.n_changed = _heat_bath_sweep(
            self.A.indptr, self.A.indices, self.A.data, self.k,
            self.sigma, self.K, order,
            self.gamma, float(beta), self.two_M, self.q, self._a, self._p)
        return self.sigma
    
    """Driver"""
    
    def anneal(self, beta_start=0.1, beta_end=10.0, n_steps=100,
               sweeps_per_step=1, verbose=False):
        """Simulated annealing with a geometric schedule in beta = 1/T.

        beta is measured in units of edge weight, so sensible values depend on
        your typical degree; check the returned energy trace and tune.

        Returns
        -------
        history : ndarray, shape (n_steps,)
            Energy after each temperature step.
        """
        betas = np.geomspace(beta_start, beta_end, n_steps)
        history = np.empty(n_steps)
        for t, b in enumerate(betas):
            for _ in range(sweeps_per_step):
                self.heat_bath_sweep(b)
            history[t] = self.hamiltonian()
            if verbose and (t % 10 == 0 or t == n_steps - 1):
                print(f"step {t:4d}  beta={b:8.3f}  H={history[t]:12.3f}  "
                      f"changed={self.n_changed:5d}  communities={self.n_communities()}")
        return history
    
    """results"""
    
    def n_communities(self):
        return len(np.unique(self.sigma))

    def communities(self):
        """List of sets of original node ids, largest community first."""
        groups = {}
        for node, s in zip(self.nodelist, self.sigma):
            groups.setdefault(int(s), set()).add(node)
        return sorted(groups.values(), key=len, reverse=True)