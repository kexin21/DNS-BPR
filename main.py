import torch
import numpy as np
import os
from parse import parse_args
import dataloader
import utils
from model import LightGCN, PureMF
import random
from time import time
from evaluation import Test_with_entropy
from helper import early_stopping
from torch.utils.tensorboard import SummaryWriter


#global variables
global args, device
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")



print(args.mess_dropout)
if args.loss == 'tau_bpr':
    writerfile = f'./log/{args.model}/{args.dataset}/{args.loss}/{args.ns}_negs_{args.n_negs}/K_{args.K}/tau_{args.tau}'
elif args.loss == 'abc_bpr':
    writerfile = f'./log/{args.model}/{args.dataset}/{args.loss}/{args.ns}_negs_{args.n_negs}/K_{args.K}/a_{args.alpha_bpr}/b_{args.beta_bpr}/c_{args.c_bpr}'
elif args.loss == 'bpr':
    writerfile = f'./log/{args.model}/{args.dataset}/{args.loss}/{args.ns}_negs_{args.n_negs}/K_{args.K}'

writer = SummaryWriter(writerfile)


#load dataset
dataset = dataloader.Loader(args, path="data/"+args.dataset)

#set recmodel
if args.model == 'lgn':
    Recmodel = LightGCN(args, dataset).to(device)
elif args.model == 'mf':
    Recmodel = PureMF(args, dataset).to(device)

#get files
weight_file, epoch_file = utils.getFileName()

#load parameters
print(f"load and save to {weight_file}")
if args.load:
    try:  
        weight = torch.load(weight_file,map_location=torch.device('cpu'))
        Recmodel.load_state_dict(weight)
        epoch_loaded = torch.load(epoch_file, map_location=torch.device('cpu'))
        print(f"loaded model weights from {weight_file}")
        print(f"starting from epoch {epoch_loaded}")
    except FileNotFoundError:
        epoch_loaded = 0
        print(f"{weight_file} not exists, start from beginning")
else:
    epoch_loaded = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        all_items = np.arange(n_items)
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            mask = ~np.in1d(all_items, train_set[user])
            noninter_items = all_items[mask]
            negitems = np.random.choice(noninter_items,n).tolist()
            # for i in range(n):  # sample n times
            #     while True:
            #         negitem = random.choice(range(n_items))
            #         if negitem not in train_set[user]:
            #             break
            #     negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end]
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    return feed_dict

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

n_users = dataset.n_users
n_items = dataset.m_items
n_negs = args.n_negs
K = args.K

train_cf_size = dataset.trainDataSize
train_cf_user = torch.LongTensor(dataset.trainUser)
train_cf_item = torch.LongTensor(dataset.trainItem)
train_cf = torch.stack([train_cf_user, train_cf_item]).t()



optimizer = torch.optim.Adam(Recmodel.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[360], gamma=0.2)


cur_best_pre_0 = 0
stopping_step = 0
should_stop = False

print("start training ...")

lower_percentile = (100 - 95) / 2
upper_percentile = 100 - lower_percentile

for epoch in range(epoch_loaded,args.epochs):
    Recmodel.epoch = epoch
    if epoch % args.evaluation_step == 0:
        """testing on val set"""

        Recmodel.eval()
        test_s_t = time()
        test_ret = Test_with_entropy(dataset, Recmodel, epoch, w=writer, mode='test')
        test_e_t = time()

        cur_best_pre_0, stopping_step, should_stop = early_stopping(test_ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc',
                                                                   flag_step=120)
        if should_stop:
            break

        """save weight"""
        if test_ret['recall'][0] == cur_best_pre_0:
            torch.save(Recmodel.state_dict(), weight_file)
            torch.save(epoch, epoch_file)

    train_cf_ = train_cf
    index = np.arange(len(train_cf_))
    np.random.shuffle(index)
    train_cf_ = train_cf_[index].to(device)

    #training
    Recmodel.train()
    loss, s = 0, 0
    train_s_t = time()

    while s + args.batch_size <= len(train_cf_):
        batch = get_feed_dict(train_cf_,
                              dataset.train_user_set,
                              s, s + args.batch_size,
                              n_negs)    

        batch_result = Recmodel(batch)
        if args.model == 'mf' or args.model == 'lgn':
          batch_loss = batch_result[0]

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss
        s += args.batch_size

    train_e_t = time()
    print('n_negs %d, using time %.4fs, training loss at epoch %d: %.4f' % (n_negs, train_e_t - train_s_t, epoch, loss.item()))
print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))