import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix,mean_squared_error
from skimage.metrics import structural_similarity as ssim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    logger,
                    config,
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()

    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images = data[:, :5, :, :]
        targets = data[:, 5:, :, :]
        images, targets = images.to(device).float(), targets.to(device).float()
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

    train_loss = np.mean(loss_list)
    train_loss = round(train_loss, 5)
    scheduler.step()
    log_info = f'Train_loss: {train_loss:.5f}'
    print(log_info)
    logger.info(log_info)
    return train_loss


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in test_loader:
            img = data[:, :5, :, :]
            msk = data[:, 5:, :, :]
            img, msk = img.to(device).float(), msk.to(device).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        for threshold in config.threshold:
            y_pre = np.where(preds >= threshold, 1, 0)
            y_true = np.where(gts >= threshold, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            HSS = (float(TP) * float(TN) - float(FN) * float(FP)) / (
                        ((float(TP) + float(FN)) * ((float(FN) + float(TN)))) + (
                            (float(TP) + float(FP)) * ((float(FP) + float(TN)))))
            POD = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            CSI = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            # print('')
            log_info = f'{threshold}:,val epoch: {epoch}, loss: {np.mean(loss_list):.5f},accuracy: {accuracy:.4f},CSI: {CSI:.4f}, HSS:{HSS:.4f}, POD: {POD:.4f}'
            # print(log_info)
            logger.info(log_info)

    else:
        # print('')
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate((test_loader)):
            img = data[:, :5, :, :]
            msk = data[:, 5:, :, :]
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()
            preds.append(out)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)
        for threshold in config.threshold:
            y_pre = np.where(preds >= threshold, 1, 0)
            y_true = np.where(gts >= threshold, 1, 0)

            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
            HSS = (float(TP) * float(TN) - float(FN) * float(FP)) / (
                    ((float(TP) + float(FN)) * ((float(FN) + float(TN)))) + (
                    (float(TP) + float(FP)) * ((float(FP) + float(TN)))))
            POD = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
            CSI = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
            RMSE = np.sqrt(mean_squared_error(gts, preds))


            # 计算结构相似性 (SSIM)
            SSIM = ssim(gts.reshape(-1), preds.reshape(-1), data_range=1)  # 假设 gts 和 preds 的形状相同

            if test_data_name is not None:
                log_info = f'test_datasets_name: {test_data_name}'
                print(log_info)
                logger.info(log_info)
            log_info = f'{threshold}:,test of best model, loss: {np.mean(loss_list):.5f}, accuracy: {accuracy:.4f},CSI: {CSI:.4f}, HSS:{HSS:.4f}, POD: {POD:.4f},SSIM: {SSIM:.4f},RMSE: {RMSE:.4f}'
            print(log_info)
            logger.info(log_info)

    return np.mean(loss_list)
