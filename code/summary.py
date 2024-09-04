import torch
from torchsummary import summary
from factory import NewModule, NewDataset
from torch_geometric.nn import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = NewModule('AMGCN').to(device)
dts = NewDataset('Deqing', True)
data = dts[0]
x = torch.randn(100, 128)
edge_index = torch.randint(100, size=(2, 20))


print(summary(model, data))