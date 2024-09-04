import torch
from dotenv import load_dotenv
import os
from factory import NewDataset, NewModule
from utils.utils import log_metrics, write_headers, calc_metrics, get_balanced_loader
from torch.utils.data import random_split
from tqdm import tqdm
from torch_geometric.loader import DataLoader

def train_loop(epoch, dataloader, model, loss_fn, optimizer, log_file, device):
    model.train()
    all_preds = []
    all_labels = []
    train_loss = 0.0
    num_batches = len(dataloader)
    
    for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
        y = batch.y.to(device)
        pred = model(batch).float().to(device)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_preds.extend((pred > 0.5).float().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    metrics = [epoch] + calc_metrics(all_preds, all_labels) + [train_loss / num_batches]
    log_metrics(metrics, log_file)

def test_loop(epoch, dataloader, model, loss_fn, log_file, device):
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
            y = batch.y.to(device).float()
            pred = model(batch).to(device)
            test_loss += loss_fn(pred, y).item()
    
            all_preds.extend((pred > 0.5).float().cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    metrics = [epoch] + calc_metrics(all_preds, all_labels) + [test_loss / num_batches]
    log_metrics(metrics, log_file)

load_dotenv()

# check if I'm in the right directory
if not os.getcwd().endswith('graduate'):
    os.chdir(os.getenv('PROJECT_PATH'))

k = int(os.getenv('K'))    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = float(os.getenv('LEARNING_RATE'))

model_name = 'SAG'
model = NewModule(model_name)
model.load_state_dict(torch.load('checkpoints/SAG_1_shot/SAG_0_20.pt'))
model = model.to(device)


checkpoint = os.path.join('checkpoints', f'{model_name}_{k}_shot/freeze_fc')
os.makedirs(checkpoint, exist_ok=True)

# for name, param in model.named_parameters():
#     print(name)
#     if not 'fc' in name:
#         param.requires_grad = False

dts = NewDataset('Deqing', True)
support_dts, query_dts = random_split(dts, [int(len(dts) * 0.3), len(dts) - int(len(dts) * 0.3)])
support_dts = get_balanced_loader(support_dts, k * 5, True)

train_log_file = os.path.join(checkpoint, f'{model_name}_train.csv')
test_log_file = os.path.join(checkpoint, f'{model_name}_test.csv')
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_dataloader = DataLoader(support_dts, shuffle=True, follow_batch=['x_q', 'x_r'])
test_dataloader = DataLoader(query_dts, shuffle=True, follow_batch=['x_q', 'x_r'])

for epoch in range(100):
    train_loop(epoch, train_dataloader, model, loss_fn, optimizer, train_log_file, device)

torch.save(model.state_dict(), os.path.join(checkpoint, f'{model_name}_{epoch + 1}.pt'))
test_loop(epoch, test_dataloader, model, loss_fn, test_log_file, device)