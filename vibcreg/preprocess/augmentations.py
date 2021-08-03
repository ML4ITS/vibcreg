"""
`Augmentations` class defines the augmentation methods.
"""
import numpy as np


class Augmentations(object):
    def __init__(self, subseq_len, AmpR_rate=0.3, Vshift_rate=0.5, **kwargs):
        """
        :param subseq_len: length of a subsequence of input time series.
        :param AmpR_rate: rate for the `random amplitude resize`.
        :param Vshift_rate: rate for the `random vertical shift`.
        """
        self.subseq_len = subseq_len
        self.AmpR_rate = AmpR_rate
        self.Vshift_rate = Vshift_rate

    def random_crop(self, *x_views):
        subx_views = []
        for i in range(len(x_views)):
            seq_len = x_views[i].shape[-1]
            subseq_len = self.subseq_len
            rand_t = np.random.randint(0, seq_len - subseq_len, size=1)[0]
            subx = x_views[i][:, rand_t: rand_t + subseq_len]  # (subseq_len)
            subx_views.append(subx)

        if len(subx_views) == 1:
            subx_views = subx_views[0]
        return subx_views

    def amplitude_resize(self, *subx_views):
        """
        :param subx_view: (n_channels * subseq_len)
        """
        new_subx_views = []
        n_channels = subx_views[0].shape[0]
        for i in range(len(subx_views)):
            mul_AmpR = 1 + np.random.uniform(-self.AmpR_rate, self.AmpR_rate, size=(n_channels, 1))
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
            new_subx_view = subx_views[i] + vshift_mag
            new_subx_views.append(new_subx_view)

        if len(new_subx_views) == 1:
            new_subx_views = new_subx_views[0]
        return new_subx_views