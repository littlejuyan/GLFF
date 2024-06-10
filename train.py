import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
sys.path.append('./data')
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, roc_curve, auc

"""Currently assumes jpg_prob, blur_prob 0 or 1"""

def validate(model, opt):
    data_loader = create_dataloader(opt)
    print("number of validation dataset: ", len(data_loader))
    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            input_img = data[0] #[batch_size, 3, height, width]
            cropped_img = data[1].cuda() #[batch_size, 3, 224, 224]
            label = data[2].cuda() #[batch_size, 1]
            scale = data[3].cuda() #[batch_size, 1, 2]

            y_pred.extend(model(input_img, cropped_img, scale).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
 
    return acc, roc_auc, ap


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)
    
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.0001, verbose=True)
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')


        print('saving the model at the end of epoch %d, iters %d' % (epoch, model.total_steps))
        model.save_networks(epoch)

        # Validation 
        model.eval()
        acc, roc_auc, ap = validate(model.model, val_opt)
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('roc_auc', roc_auc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; roc_auc: {}; ap: {}".format(epoch, acc, roc_auc, ap))
        info = [str(epoch), ',', str(acc), ',', str(roc_auc), ',', str(ap)]
        with open('./evalacc.txt', 'a') as f:
            f.writelines(info)
            f.writelines('\n')
        
        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 2, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.00005, verbose=True)
            else:
                print("Learning rate dropped to minimum, still training with minimum learning rate...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.00005, verbose=True)
                break

        model.train()
