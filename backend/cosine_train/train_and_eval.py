"""
Train & Eval standalone script for cosine metric learning
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from backend.utils import Log

import os
import math
import torch.nn as nn

from backend.cosine_train.dataset import CropTrainDataset, CropTestDataset
from backend.cosine_metric_net import CosineMetricNet
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from backend.cosine_train.metric import mAPMetric
import warnings
warnings.filterwarnings("ignore")  # to ignore UserWarning


class Config:
    resume = True
    resume_ckpt = './ckpts/model0.pt'

    train_dir = "./crops"
    test_dir = "./crops_test/"
    model_dir = "./ckpts/"
    train_batch_size = 350
    test_batch_size = 32
    train_number_epochs = 500
    stats_iter_frequency = 10
    checkpoint_frequency = 5
    random_seed = 233


if __name__ == '__main__':
    train_dataset = datasets.ImageFolder(root=Config.train_dir)
    test_dataset = datasets.ImageFolder(root=Config.test_dir)
    X, Y = zip(*train_dataset.imgs)
    X_test, Y_test = zip(*test_dataset.imgs)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=Config.random_seed, stratify=Y)
    train_dataset_ins = CropTrainDataset(X, Y)
    test_dataset_ins = CropTestDataset(X_test, Y_test)

    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()  # for CosineMetricLearning

    model = CosineMetricNet(num_classes=len(train_dataset.class_to_idx)).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    st_epoch_num = 0
    if Config.resume and len(Config.resume_ckpt) != 0:
        Log.info("Resume from checkpoint {}...".format(Config.resume_ckpt))
        ckpt = torch.load(Config.resume_ckpt)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        st_epoch_num = ckpt['epoch']

    # model = torch.load('ckpts/model405.pt')
    train_dataloader = DataLoader(train_dataset_ins, batch_size=Config.train_batch_size,
                                  shuffle=True, num_workers=12, drop_last=True)
    test_dataloader = DataLoader(test_dataset_ins, batch_size=Config.test_batch_size, shuffle=False,
                                 num_workers=12, drop_last=False)
    map_validator = mAPMetric()

    dummy_input = torch.randn(10, 3, 128, 64).cuda()  # random input for test
    writer.add_graph(model, (dummy_input,))

    tot_iter = 0

    for i in range(st_epoch_num, Config.train_number_epochs):
        model.train()
        for j, batched_data in enumerate(train_dataloader):
            img, img_labels = batched_data
            img, img_labels = Variable(img).cuda(), Variable(img_labels).cuda()
            optimizer.zero_grad()
            feature, logits = model(img)  # feature: [N, 128]   logits: [N, num_classes]
            loss = criterion(logits, img_labels)
            loss.backward()
            optimizer.step()
            tot_iter += 1
            if j % Config.stats_iter_frequency == 0:
                train_mAP = map_validator.get_value(logits.cpu(), img_labels.cpu().unsqueeze(1)).item()
                Log.info("Epoch %d, Iter %d, Current cosine-softmax Loss %.5f, train_mAP %.3f" % (i, j + 1, loss.item(), train_mAP))
                writer.add_scalar('train_cosine_loss', loss.item(), tot_iter)
                writer.add_scalar('train_mAP', train_mAP, tot_iter)

            # anchor, positive, negative = batched_data  # type: Variable, Variable, Variable
            # anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            # optimizer.zero_grad()
            # anchor_out, positive_out, negative_out = model(anchor, positive, negative)
            # loss = criterion(anchor_out, positive_out, negative_out)
            # loss.backward()
            # optimizer.step()
            # if j % Config.stats_iter_frequency == 0:
            #     Log.info("Epoch %d, Iter %d, Current Triplet Loss %.5f" % (i, j+1, loss.item()))

        Log.info("Epoch %d Done, Start Eval..." % i)

        model.eval()
        batch_correct = 0
        batch_num = 0
        tot_map, tot_loss = 0.0, 0.0
        for k, batched_data in enumerate(test_dataloader):
            batch_num += 1
            anchor, anchor_class = batched_data
            # anchor, anchor_class = batched_data
            anchor = Variable(anchor).cuda()
            # positive = Variable(positive).cuda()
            anchor_class = Variable(anchor_class).cuda()

            anchor_out, logits = model(anchor)
            eval_loss = criterion(logits, anchor_class)

            tot_loss += eval_loss.item()
            single_map = map_validator.get_value(logits.cpu(), anchor_class.cpu().unsqueeze(1)).item()
            tot_map += single_map
            print("current batch: " + str(k + 1) + '/' + str(
                math.ceil(len(test_dataset_ins) / Config.test_batch_size)) + ", eval loss: " + str(eval_loss.item()) + 'eval map: ' + str(single_map))

        Log.info("Batch mAP: %.3f" % (tot_map / batch_num))
        Log.info("Batch Eval Loss: %.3f" % (tot_loss / batch_num))
        writer.add_scalar('eval_cosine_loss', (tot_loss / batch_num), i + 1)
        writer.add_scalar('batch_mAP', (tot_map / batch_num), i + 1)
        # DONE: mAP validator
        # TODO: CMC validator needed in the future
        # TODO: currently we just compare cosine similarity metric

        if i % Config.checkpoint_frequency == 0:
            if not os.path.exists(Config.model_dir):
                os.mkdir(Config.model_dir)
            ckpt_dict = {
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(ckpt_dict, os.path.join(Config.model_dir, 'model' + str(i) + '.pt'))
