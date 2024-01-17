from .base import MetricsCalculatorBase
from ..core import PipelineError

from sklearn import metrics
import numpy as np

class MetricsCalculatorAuroc(MetricsCalculatorBase):
    def __init__(self):
        super().__init__()
        self.zero_cache()

    def zero_cache(self):
        self._predictions = []
        self._true_labels = []

    def add(self, y_predicted, y_true):
        self._predictions.append(y_predicted.flatten())
        self._true_labels.append(y_true.flatten())
       
    def calculate(self):
        if not self._predictions:
            raise PipelineError("You need to add predictions for calculating the accuracy first")

        y_pred = np.concatenate(self._predictions)
        y_true = np.concatenate(self._true_labels)

        fpr, tpr, tresh = metrics.roc_curve(y_true,y_pred)
        auc = metrics.auc(fpr, tpr)

        return {"auc": auc}
