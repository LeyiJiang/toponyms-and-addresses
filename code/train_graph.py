import os
from dotenv import load_dotenv
import tqdm
import factory
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from utils.utils import log_metrics, write_headers, calc_metrics

def train_loop(epoch, dataloader, model, loss_fn, optimizer, log_file, device):
    model.train()
    all_preds = []
    all_labels = []
    train_loss = 0.0
    num_batches = len(dataloader)
    
    for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}'):
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
        for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}'):
            y = batch.y.to(device).float()
            pred = model(batch).to(device)
            test_loss += loss_fn(pred, y).item()
    
            all_preds.extend((pred > 0.5).float().cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    metrics = [epoch] + calc_metrics(all_preds, all_labels) + [test_loss / num_batches]
    log_metrics(metrics, log_file)

model_name = 'GCN'

load_dotenv()
batch_size = int(os.getenv('BATCH_SIZE'))
epochs = 10
learning_rate = float(os.getenv('LEARNING_RATE'))
experiment_time = int(os.getenv('EXPERIMENT_TIME'))
k = int(os.getenv('K'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# check if I'm in the right directory
if not os.getcwd().endswith('graduate'):
    os.chdir(os.getenv('PROJECT_PATH'))

checkpoint = os.path.join('checkpoints', f'{model_name}_{k}_shot')
os.makedirs(checkpoint, exist_ok=True)

for round in range(experiment_time):
    model = factory.NewModule(model_name)
    train_set = factory.NewDataset('Tianchi', True)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, follow_batch=['x_q', 'x_r'])

    dts = factory.NewDataset('Deqing', True)
    validation_dts, query_dts = random_split(dts, [int(len(dts) * 0.3), len(dts) - int(len(dts) * 0.3)])

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_log_file = os.path.join(checkpoint, f'{model_name}_{round}_train.csv')
    write_headers(['epoch', 'accuracy', 'recall', 'precision', 'f1', 'loss'], train_log_file)

    for epoch in range(epochs):
        train_loop(epoch, train_dataloader, model, loss_fn, optimizer, train_log_file, device)
        if ( epoch + 1 ) % 5 == 0:
            # test_loop(epoch, test_dataloader, model, loss_fn, test_log_file, device)
            torch.save(model.state_dict(), os.path.join(checkpoint, f'{model_name}_{round}_{epoch + 1}.pt'))

print('Done!')