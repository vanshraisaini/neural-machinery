import torch


class GradientDescent:
    def __init__(self, params, alpha=0.01):
        self.params = params
        self.alpha = alpha

    def step(self):
        with torch.no_grad():
            for param in self.params:
                if(param.grad is not None):
                    param -= self.alpha * param.grad

    def zero_grad(self):
        for param in self.params:
            if(param.grad is not None):
                param.grad.zero_()