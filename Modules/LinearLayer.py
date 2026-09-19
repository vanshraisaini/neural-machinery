import torch


class LinearLayer:
    def __init__(self, input_dim, output_dim, bias=True):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.weights = torch.randn((input_dim, output_dim), dtype=torch.float64, requires_grad=True)
        if(bias):
            self.biases = torch.randn((1, output_dim), dtype=torch.float64, requires_grad=True)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        #inputs: (B, D); weights: (D,HD); Biases: (1, HD)
        output = inputs @ self.weights
        if(self.bias):
            output += self.biases
        return output

    def get_params(self):
        if(self.biases):
            return [self.weights, self.biases]
        else:
            return [self.weights]


if __name__ == "__main__":
    layer = LinearLayer(5,10)
    print(layer.weights.shape)
    print(layer.biases.shape)
    print(layer.forward(torch.rand((12,5), dtype=torch.float64)).shape)