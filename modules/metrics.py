import torch


class Metrics:
    def __init__(self):
        self.name = "Metric Name"

    def reset(self):
        pass

    def update(self, predicts, targets):
        pass

    def get_score(self):
        pass


class MulticlassAccuracy(Metrics):
    """ Multiclass Classification Accuracy
    """

    def __init__(self):
        self.name = "Acc."
        self.n_correct = 0
        self.n = 1e-20

    def reset(self):
        self.n_correct = 0
        self.n = 1e-20

    def update(self, predicts, targets):
        predicts = torch.exp(predicts).max(dim=1)[1]
        self.n_correct += (predicts == targets).sum().item()
        self.n += targets.shape[0]

    def get_score(self):
        return self.n_correct / self.n

    def print_score(self):
        score = self.get_score()
        return "{:.5f}".format(score)
