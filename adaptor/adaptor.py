from constants import NORM_TYPES
import datasets



class Adaptor():

    def __init__(self, dataset, model):
        assert dataset in datasets.DATASETS
        self.dataset = dataset
        self.model = model

    def verify(self, input, label, norm_type, radius) -> bool:
        """
            The default implementation assumes the completeness of calc_radius() function
        :param input: input sample
        :param label: correct label
        :param norm_type: norm type
        :param radius: the floating point radius
        :return: bool
        """
        r = self.calc_radius(input, label, norm_type)
        return radius <= r

    def calc_radius(self, input, label, norm_type, upper=0.5, eps=1e-2) -> float:
        """
            The default implementation assumes the completeness of verify() function
        :param input: input sample
        :param label: correct label
        :param norm_type: norm type
        :return: the floating point radius
        """
        assert norm_type in NORM_TYPES
        if not self.verify(input, label, norm_type, 0.0):
            return 0.0
        l = 0.0
        r = upper
        while r-l > eps:
            # print(l, r)
            mid = (l + r) / 2.0
            if self.verify(input, label, norm_type, mid):
                l = mid
            else:
                r = mid
        return l



