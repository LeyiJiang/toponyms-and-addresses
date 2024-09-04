import os
from dotenv import load_dotenv
import tqdm
import factory
import torch
from torch.utils.data import random_split, DataLoader
from utils.utils import log_metrics, write_headers, calc_metrics, get_balanced_loader

def train_loop(epoch, dataloader, model, loss_fn, optimizer, log_file, device):
    model.train()
    all_preds = []
    all_labels = []
    train_loss = 0.0
    num_batches = len(dataloader)

    for Q, R, y in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}'):
        Q, R, y = Q.to(device), R.to(device), y.to(device)
        pred = model(Q, R).to(device)
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
        for Q, R, y in dataloader:
            Q, R, y = Q.to(device), R.to(device), y.to(device)
            pred = model(Q, R).to(device)
            test_loss += loss_fn(pred, y).item()
    
            all_preds.extend((pred > 0.5).float().cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    metrics = [epoch] + calc_metrics(all_preds, all_labels) + [test_loss / num_batches]
    log_metrics(metrics, log_file)


load_dotenv()
batch_size = int(os.getenv('BATCH_SIZE'))
epochs = int(os.getenv('EPOCHS'))
learning_rate = float(os.getenv('LEARNING_RATE'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = ['SiaGRU']

for idx in range(len(models)):
    model_name = models[idx]
    # check if I'm in the right directory
    if not os.getcwd().endswith('graduate'):
        os.chdir(os.getenv('PROJECT_PATH'))

    checkpoint = os.path.join('checkpoints', f'{model_name}')
    os.makedirs(checkpoint, exist_ok=True)

    model = factory.NewModule(model_name).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_set = factory.NewDataset('Tianchi', False)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    train_log_file = os.path.join(checkpoint, f'{model_name}_train.csv')
    write_headers(['epoch', 'accuracy', 'recall', 'precision', 'f1', 'loss'], train_log_file)

    for epoch in range(epochs):
        train_loop(epoch, train_dataloader, model, loss_fn, optimizer, train_log_file, device)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint, f'{model_name}_{epoch + 1}_epoch.pt'))

    ks = [0, 5, 10, 50, 100]
    for round in range(len(ks)):
        k = ks[round]
        if not os.getcwd().endswith('graduate'):
            os.chdir(os.getenv('PROJECT_PATH'))
        checkpoint = os.path.join('checkpoints', f'{model_name}_{k}_shot')
        os.makedirs(checkpoint, exist_ok=True)
        dts = factory.NewDataset('Deqing', False)
        support_dts, query_dts = random_split(dts, [int(len(dts) * 0.3), len(dts) - int(len(dts) * 0.3)])
        if k > 0:
            support_dts = get_balanced_loader(support_dts, k)

            train_log_file = os.path.join(checkpoint, f'{model_name}_train.csv')
            train_dataloader = DataLoader(support_dts, batch_size=k, shuffle=True)
            write_headers(['epoch', 'accuracy', 'recall', 'precision', 'f1', 'loss'], train_log_file)
            for epoch in range(100):
                train_loop(epoch, train_dataloader, model, loss_fn, optimizer, train_log_file, device)

        test_log_file = os.path.join(checkpoint, f'{model_name}_test.csv')
        test_dataloader = DataLoader(query_dts, batch_size=128, shuffle=True)
        write_headers(['epoch', 'accuracy', 'recall', 'precision', 'f1', 'loss'], test_log_file)
        epoch = 99
        torch.save(model.state_dict(), os.path.join(checkpoint, f'{model_name}_{epoch+1}.pt'))
        test_loop(epoch, test_dataloader, model, loss_fn, test_log_file, device)

print('Done!')