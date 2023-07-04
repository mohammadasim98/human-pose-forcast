import torch

class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k

    def compute(self, output, target):
        
        correct = 0

        with torch.no_grad():

            _, pred = torch.topk(output, self.k, dim=1)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct = correct[:self.k].float().sum()

        return correct / len(target)
    
    def __str__(self):
        return f"top{self.k}"