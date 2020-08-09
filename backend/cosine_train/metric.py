from torchnet.meter import mAPMeter
import torch


class BaseMetric(object):
    def __init__(self):
        pass

    def get_value(self, output: torch.Tensor, target: torch.Tensor):
        return NotImplemented

    def __call__(self, *args, **kwargs):
        return self.get_value(*args)


class mAPMetric(BaseMetric):
    """
    A custom wrapper for torchnet.meter.mAPMeter
    """

    def __init__(self):
        super(mAPMetric, self).__init__()
        self.m_ap = mAPMeter()

    @staticmethod
    def _idx2one_hot(input: torch.Tensor, class_num: int):
        """
        :param input: Tensor of size (N,1)
        :param class_num:
        :return: Tensor of size (N, class_num)
        """
        return torch.zeros(input.shape[0], class_num, dtype=torch.bool).scatter_(1, input, 1)

    def get_value(self, output: torch.Tensor, target: torch.Tensor):
        """
        :param output: Tensor of size (N,K) for N examples and K classes
        :param target: Tensor of size (N,1), should be one-hot encoded before submitting into mAPMeter
        :return:
        """
        self.m_ap.reset()
        target = self._idx2one_hot(target, output.shape[1]).int()
        self.m_ap.add(output, target)
        return self.m_ap.value()


class CMCMetric(BaseMetric):
    """
    Waiting to be implemented
    """
    def __init__(self):
        super(CMCMetric, self).__init__()

    def get_value(self, output: torch.Tensor, target: torch.Tensor):
        pass
