import numpy as np

###############################################################################
# OPTIONAL: If you want to use KDE or GMM, you'll need scikit-learn installed:
#     pip install scikit-learn
###############################################################################

class MDCDFAggregator:
    def __init__(
        self,
        method='kde',           # 'kde' or 'gmm' or 'dummy'
        invert_cdf=False,
        debug_enabled=True,
        **model_kwargs
    ):
        """
        A multi-dimensional aggregator that:
          1) Fits a density model (KDE or GMM) on the ID data's OOD-score vectors.
          2) At inference, we compute each test sample’s log-likelihood,
             then rank it against the ID log-likelihood distribution => a (0..1) CDF.

        Arguments:
          - method: 'kde', 'gmm', or 'dummy' (dummy means fallback to the old “sigmoid+mean”).
          - invert_cdf: whether to invert the final score => final_score = 1 - cdf.
          - debug_enabled: print debug logs if True.
          - model_kwargs: passed to sklearn's KernelDensity or GaussianMixture, e.g. `bandwidth=0.5` or `n_components=2`.
        """
        self.method = method
        self.invert_cdf = invert_cdf
        self.debug_enabled = debug_enabled
        self.model_kwargs = model_kwargs

        self.id_data = None
        self.kde = None
        self.gmm = None

    def fit(self, data_2d: np.ndarray):
        """
        Fit the aggregator on the ID-layer OOD scores, shape [N, D].
        For 'kde' => fit a KernelDensity model,
        For 'gmm' => fit a GaussianMixture,
        For 'dummy' => just store data.
        """
        if self.debug_enabled:
            print(f"[MDCDFAggregator] Fit: Storing {len(data_2d)} ID vectors in {data_2d.shape[1]}-dim space.")

        self.id_data = data_2d

        if self.method == 'kde':
            from sklearn.neighbors import KernelDensity
            self.kde = KernelDensity(**self.model_kwargs)
            self.kde.fit(data_2d)
            if self.debug_enabled:
                print(f"[MDCDFAggregator] KDE model fitted with params={self.model_kwargs}")

        elif self.method == 'gmm':
            from sklearn.mixture import GaussianMixture
            self.gmm = GaussianMixture(**self.model_kwargs)
            self.gmm.fit(data_2d)
            if self.debug_enabled:
                print(f"[MDCDFAggregator] GMM model fitted with params={self.model_kwargs}")

        else:
            # dummy method, do nothing
            if self.debug_enabled:
                print("[MDCDFAggregator] 'dummy' method => no density model, fallback to old approach.")

        if self.debug_enabled and len(data_2d) > 0:
            mn, mx, me = data_2d.min(), data_2d.max(), data_2d.mean()
            print(f"[MDCDFAggregator] Example ID data stats => min: {mn} max: {mx} mean: {me:.4f}")

    def score(self, x_2d: np.ndarray) -> np.ndarray:
        """
        x_2d => shape [M, D], produce aggregator output in [0,1].

        If 'kde'/'gmm':
          1) Compute log-likelihood for x_2d: shape [M].
          2) Also have log-likelihood for ID data => shape [N].
          3) The aggregator score for x_2d[i]:
                 cdf = (# of ID samples with loglike <= loglike_x_i) / N
             This is a simple rank-based approach.

        If 'dummy':
          fallback => cdf_vals = mean( sigmoid(x_2d) ) across dims => shape [M].
        """
        M, D = x_2d.shape
        if self.debug_enabled and M > 0:
            print(f"[MDCDFAggregator] score() called on shape {x_2d.shape}")
            print("[MDCDFAggregator] first row =>", x_2d[0])

        # -------------- KERNEL DENSITY OR GMM -------------- #
        if self.method in ['kde', 'gmm']:
            if self.id_data is None:
                raise ValueError("[MDCDFAggregator] No ID data stored. Did you call fit?")

            # 1) log-likelihood for ID data
            if self.method == 'kde':
                ll_id = self.kde.score_samples(self.id_data)    # shape [N]
                ll_x  = self.kde.score_samples(x_2d)            # shape [M]
            else:  # 'gmm'
                ll_id = self.gmm.score_samples(self.id_data)
                ll_x  = self.gmm.score_samples(x_2d)

            # 2) sort ID log-likelihood
            ll_id_sorted = np.sort(ll_id)  # shape [N]

            # 3) for each x, find fraction
            N = len(ll_id)
            cdf_vals = []
            for val in ll_x:
                # fraction = (# of ID loglike <= val) / N
                # We can do a binary search with np.searchsorted
                count = np.searchsorted(ll_id_sorted, val, side='right')
                frac = count / N
                cdf_vals.append(frac)
            cdf_vals = np.array(cdf_vals)

            # optionally invert
            if self.invert_cdf:
                cdf_vals = 1 - cdf_vals

        # -------------- FALLBACK: DUMMY -------------- #
        else:
            # Just do the old 'sigmoid + mean' across dims
            # For demonstration only
            cdf_vals = 1/(1+np.exp(-x_2d))  # shape [M, D]
            cdf_vals = cdf_vals.mean(axis=1)  # shape [M]
            if self.invert_cdf:
                cdf_vals = 1 - cdf_vals

        if self.debug_enabled and M>0:
            print("[MDCDFAggregator] first aggregator output =>", cdf_vals[0])

        return cdf_vals
