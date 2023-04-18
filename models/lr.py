import torch

class LinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(torch.flatten(x, start_dim=1))

    def getPLR(self, x):
        return self.forward(x)

if __name__ == "__main__":
    net = LinearRegression(input_dim=10, output_dim=10)
    net.eval()
    torch.manual_seed(1)
    x   = torch.randn(10, 10)
    out = net(x)
    print("len(out):", len(out))
    print("out.shape:", out.shape)