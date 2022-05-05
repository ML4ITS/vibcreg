"""
`Augmentations` class defines the augmentation methods.
"""
import numpy as np


class Augmentations(object):
    def __init__(self, subseq_len: int, AmpR_rate=0.3, Vshift_rate=0.5, **kwargs):
        """
        :param subseq_len: length of a subsequence of input time series for random crop.
        :param AmpR_rate: rate for the `random amplitude resize`.
        :param Vshift_rate: rate for the `random vertical shift`.
        """
        self.subseq_len = subseq_len
        self.AmpR_rate = AmpR_rate
        self.Vshift_rate = Vshift_rate

        self.rc_dist = None

    def random_crop(self, *x_views):
        subx_views = []
        rand_ts = []
        for i in range(len(x_views)):
            seq_len = x_views[i].shape[-1]
            subseq_len = self.subseq_len
            rand_t = np.random.randint(0, seq_len - subseq_len + 1, size=1)[0]
            subx = x_views[i][:, rand_t: rand_t + subseq_len]  # (subseq_len)
            subx_views.append(subx)
            rand_ts.append(rand_t)

        if len(rand_ts) == 2:
            self.rc_dist = np.abs(rand_ts[1] - rand_ts[0]).astype(float)
            self.rc_dist = min(self.rc_dist, self.subseq_len) / self.subseq_len

        if len(subx_views) == 1:
            subx_views = subx_views[0]
        return subx_views

    def neigh_random_crop(self, x):

        # reference sample
        subseq_len = self.subseq_len
        seq_len = x.shape[-1]

        if subseq_len == seq_len:
            return x, x, x
        else:
            # neigh samples
            neigh_rng = subseq_len//2
            while True:
                t1 = np.random.randint(subseq_len // 2, seq_len - subseq_len // 2)
                mu, sigma = t1, np.sqrt(subseq_len/2/2)  # 1st `/2`: "radius-arm"; 2nd `/2`: to shorten the neighborhood range.
                t2 = mu + int(sigma * np.random.randn())
                if (t2-subseq_len//2 >= 0) and (t2+subseq_len//2 <= seq_len):
                    break
            x1 = x[:, t1 - subseq_len // 2: t1 + subseq_len // 2]  # (subseq_len,)
            x2 = x[:, t2-subseq_len//2: t2+subseq_len//2]

            # non-neigh samples
            while True:
                t3 = np.random.randint(subseq_len//2, seq_len - subseq_len//2)
                if (t3 <= t1 - neigh_rng) or (t3 >= t1 + neigh_rng):
                    if (t3 - subseq_len // 2 >= 0) and (t3 + subseq_len // 2 <= seq_len):
                        break
            x3 = x[:, t3 - subseq_len // 2: t3 + subseq_len // 2]

            return x1, x2, x3

    def amplitude_resize(self, *subx_views):
        """
        :param subx_view: (n_channels * subseq_len)
        """
        new_subx_views = []
        n_channels = subx_views[0].shape[0]
        for i in range(len(subx_views)):
            mul_AmpR = 1 + np.random.uniform(-self.AmpR_rate, self.AmpR_rate, size=(n_channels, 1))
            # mul_AmpR = 1 + np.random.normal(0, self.AmpR_rate, size=(n_channels, 1))
            new_subx_view = subx_views[i] * mul_AmpR
            new_subx_views.append(new_subx_view)

        if len(new_subx_views) == 1:
            new_subx_views = new_subx_views[0]
        return new_subx_views

    def vertical_shift(self, std_x, *subx_views):
        new_subx_views = []
        for i in range(len(subx_views)):
            vshift_mag = self.Vshift_rate * std_x
            vshift_mag = np.random.uniform(-vshift_mag, vshift_mag)
            # vshift_mag = np.random.normal(0, vshift_mag)
            new_subx_view = subx_views[i] + vshift_mag
            new_subx_views.append(new_subx_view)

        if len(new_subx_views) == 1:
            new_subx_views = new_subx_views[0]
        return new_subx_views