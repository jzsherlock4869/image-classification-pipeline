# Top1 acc, Top 5 acc

from locale import normalize
import numpy as np
from sklearn.metrics import top_k_accuracy_score, confusion_matrix

# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from utils.meter_utils import AverageMeter

class MetricTopKAcc:
    """
        metric acc@k, e.g. Top1 acc, Top5 acc
    """
    def __init__(self, k=1) -> None:
        self.lbls = []
        self.preds = []
        self.k = k

    def update(self, lbl, pred):
        # lbl: Tensor size [n, 1], e.g. 2 (2-th class)
        # pred: prob vec size [n, c], e.g. [0.1, 0.3, 0.6] (top 1 is 2-th class)
        bs = lbl.size()[0]
        pred_np = pred.detach().cpu().numpy()
        for i in range(bs):
            self.lbls.append(lbl[i].item())
            self.preds.append(pred_np[i])
    
    def num_sample(self):
        return len(self.lbls)

    def reset(self):
        self.lbls = []
        self.preds = []

    def calc(self):
        acc = top_k_accuracy_score(np.array(self.lbls), np.array(self.preds), k=self.k, normalize=True)
        return acc

    def calc_confusion_mat(self):
        confusion_mat = confusion_matrix(np.array(self.lbls), np.array(self.preds).argmax(axis=1))
        return confusion_mat

