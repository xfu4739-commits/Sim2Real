
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import torch
import torch.distributed as dist


class BinaryAUPRCHistogram:
    """Memory-bounded, distributed binary average precision.

    Scores are accumulated into fixed probability bins. The two histograms can
    be summed across DDP ranks, unlike averaging per-batch AP values, which is
    mathematically incorrect. With 65,536 bins the approximation error is
    negligible for float32 segmentation probabilities while memory stays fixed.
    """

    def __init__(self, device, bins=65536):
        if bins < 2:
            raise ValueError("bins must be at least 2")
        self.bins = int(bins)
        self.device = torch.device(device)
        self.positive = torch.zeros(self.bins, dtype=torch.float64, device=self.device)
        self.negative = torch.zeros(self.bins, dtype=torch.float64, device=self.device)

    def update(self, scores, targets):
        scores = scores.detach().to(self.device).reshape(-1).clamp(0, 1)
        targets = targets.detach().to(self.device).reshape(-1).bool()
        indices = torch.clamp(
            (scores * (self.bins - 1)).long(),
            min=0,
            max=self.bins - 1,
        )
        self.positive += torch.bincount(
            indices[targets], minlength=self.bins
        ).to(torch.float64)
        self.negative += torch.bincount(
            indices[~targets], minlength=self.bins
        ).to(torch.float64)

    def compute(self, sync_dist=False):
        positive = self.positive.clone()
        negative = self.negative.clone()
        if sync_dist and dist.is_available() and dist.is_initialized():
            dist.all_reduce(positive, op=dist.ReduceOp.SUM)
            dist.all_reduce(negative, op=dist.ReduceOp.SUM)

        true_positive = torch.flip(positive, dims=(0,)).cumsum(0)
        false_positive = torch.flip(negative, dims=(0,)).cumsum(0)
        total_positive = true_positive[-1]
        if total_positive <= 0:
            return torch.tensor(0.0, dtype=torch.float64, device=self.device)

        precision = true_positive / (true_positive + false_positive).clamp_min(1)
        recall = true_positive / total_positive
        previous_recall = torch.cat([torch.zeros_like(recall[:1]), recall[:-1]])
        return ((recall - previous_recall) * precision).sum()


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2) 

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  
        return meanAcc  

    def IntersectionOverUnion(self):
        intersection = np.diag(self.confusionMatrix) 
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  
        IoU = intersection / union  
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion()) 
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel): 
        """
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def f1(self):
        TP = self.confusionMatrix[1, 1]  
        FP = self.confusionMatrix[0, 1]  
        FN = self.confusionMatrix[1, 0]  
        TN = self.confusionMatrix[0, 0]  

        Precision = TP / (TP + FP) if TP + FP > 0 else 0
        Recall = TP / (TP + FN) if TP + FN > 0 else 0
        F1 = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall > 0 else 0
        return F1

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

