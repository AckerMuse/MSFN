from math import log10


def PSNR(mse, peak=1.):
	return 10 * log10((peak ** 2) / mse)


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)#将one-hot概率转化为0、1
    correct = preds.eq(labels).double()#判断预测值与lables是否相等
    correct = correct.sum()#统计预测正确的样本个数
    return correct / len(labels)