import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch.optim as optim
import torch
from metrics import get_cindex
from dataset import *
from MSNmodel import MSN_DTA
from utils import *
from log.train_logger import TrainLogger
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


##########################

Data_root = "../data",
save_dir = "save",
Dataset = 'davis',
save_model = '../code/output/model',
load_model_path = None
NUM_EPOCHS = 300
steps_per_epoch = 50

######################################

params = dict(
    data_root="../data",
    save_dir="save",
    dataset='davis',
    save_model='../code/output/model',
    # dataset=args.dataset,
    # save_model=args.save_model,
    lr=5e-4,
    batch_size=128
)

logger = TrainLogger(params)
logger.info(__file__)


def main():

    fpath = os.path.join("../data", 'davis')
    train_set = PreDataset(fpath, train=True)
    test_set = PreDataset(fpath, train=False)
    break_flag = False
    PreTrain = False
    early_stop = 50
    reset_epochs = 50

    print('The train data number is {}'.format(len(train_set)))
    print('The test data number is {}'.format(len(test_set)))

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=1)

    device = torch.device('cuda:0')

    model = MSN_DTA(3, 22, 25 + 1, embedding_size=128, filter_num=32, out_dim=1, blpretrain=PreTrain).to(device)

    if load_model_path is not None:
        print('load_model...', load_model_path)
        load_model = torch.load(load_model_path)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in load_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr = 5e-4)
    criterion = nn.MSELoss()

    run_MSE = AverageMeter()
    run_cindex = AverageMeter()
    run_best_MSE = BestMeter('min')

    model.train()

    for i in range(NUM_EPOCHS):

        if break_flag:
            break

        for data in tqdm(train_loader):

            data = data.to(device)
            pred = model(data, i + 1, reset_epochs)
            mse_loss = criterion(pred.view(-1), data.y.view(-1))
            cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            run_MSE.update(mse_loss.item(), data.y.size(0))
            run_cindex.update(cindex, data.y.size(0))

        epoch_mse = run_MSE.get_average()
        epoch_index = run_cindex.get_average()
        run_MSE.reset()
        run_cindex.reset()
        if (i + 1) % reset_epochs == 0:
            print("Update node feature!")
            continue
        else:
            test_loss = val(model, criterion, test_loader, device)

            message = "Epoch-%d, MSE-%.4f, Cindex-%.4f, Test_MSE-%.4f" % (i, epoch_mse, epoch_index, test_loss)
            logger.info(message)

            if test_loss <run_best_MSE.get_best():
                run_best_MSE.update(test_loss)
                if save_model:
                    save_model_dict(model, logger.get_model_dir(), message)
            else:
                count = run_best_MSE.counter()
                if count > early_stop:
                    logger.info(f"early stop in epoch {i}")
                    break_flag = True


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data, 1, 9999)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss


if __name__ == "__main__":
    main()
