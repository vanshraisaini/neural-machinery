import torch


class ReLU:
    def __init__(self):
        pass
    
    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        #inputs: (B, D)
        mask = inputs<0
        return torch.where(mask, 0, inputs)


if __name__ == "__main__":
    relu = ReLU()
    print(relu(torch.tensor([[-1,2,-2]])))