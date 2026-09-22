import torch


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        #inputs: (B, D)
        mask = inputs<0
        return torch.where(mask, self.alpha*inputs, inputs)


if __name__ == "__main__":
    relu = LeakyReLU(alpha=0.01)
    print(relu(torch.tensor([[-1,2,-2]])))