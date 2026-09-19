import torch


class MSE:
    def __init__(self):
        pass

    def __call__(self, gt, pred):
        return self.calculate_loss(gt, pred)
    
    def calculate_loss(self, ground_truth, predicted):
        #(B, D)
        return torch.mean((ground_truth-predicted)**2)


if __name__ == "__main__":
    mse = MSE()
    print(mse(torch.tensor([1.,2,3]), torch.tensor([1.,2,5])))