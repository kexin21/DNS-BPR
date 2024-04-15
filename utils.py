from parse import parse_args
import torch
from torch import nn
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from sklearn.metrics import roc_auc_score
import random
import os
from math import log, e
from os.path import join, dirname


args = parse_args()

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

try:
    from cppimport import imp_from_filepath
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(args.seed)
    sample_ext = True
except:
    cprint("Cpp extension not loaded")
    sample_ext = False


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(ROOT_PATH, 'checkpoints')

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

def samplebatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', args.batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        i = range(0, len(tensors[0]), batch_size)[2]
        return tensor[i:i + batch_size]
    else:
        i = range(0, len(tensors[0]), batch_size)[2]
        return tuple(x[i:i + batch_size] for x in tensors)


def UniformSample_original_python_efficient(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    users = dataset.trainUser

    negitems_candidates = np.random.choice(dataset.m_items, [len(users), 1])
    for i in range(len(users)):
        while negitems_candidates[i, 0] in dataset.allPos[users[i]]:
            negitems_candidates[i, 0] = random.randint(0, dataset.m_items - 1)
    users = np.reshape(users, [-1,1])
    negitems = np.reshape(negitems_candidates, [-1, 1])
    positems = np.reshape(np.array(dataset.trainItem), [-1, 1])

    S = np.hstack([users,positems,negitems])
    total = time() - total_start

    return S




def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    if args.loss == 'abc_bpr':
        dir_path = os.path.join(FILE_PATH,f"{args.model}-{args.dataset}-F{args.recdim}-{args.ns}-{args.loss}-n_negs{args.n_negs}-a{args.alpha_bpr}-b{args.beta_bpr}-c{args.c_bpr}-lr{args.lr}-K{args.K}")
    elif args.loss == 'bpr':
        dir_path = os.path.join(FILE_PATH,f"{args.model}-{args.dataset}-F{args.recdim}-{args.ns}-{args.loss}-n_negs{args.n_negs}-lr{args.lr}-K{args.K}")
  
    isexist = os.path.exists(dir_path)
    if not isexist:
        os.makedirs(dir_path)
    File = os.path.join(dir_path, "parameter.pth.tar")
    epoch_file = os.path.join(dir_path, "epoch")
    return File, epoch_file

def entropy1(labels, base=None):
    try:
      n_labels = len(labels)
    except:
      n_labels = 1

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', args.batch_size)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def ENTROPY_ATk(pred_item, k, category_label):
    pred_k = pred_item[:,:k]
    cat_k = category_label[pred_k]
    entropy = np.array([entropy1(cat_k[i]) for i in range(cat_k.shape[0])])

    entropy = np.sum(entropy)
    return entropy

def COVERAGE(pred_item, k, category_label):
    pred_k = pred_item[:,:k]
    cat_k = category_label[pred_k]
    coverage = np.array([len(np.unique(cat_k[i])) for i in range(cat_k.shape[0])])

    coverage = np.sum(coverage)
    return coverage


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def getLabel_with_entropy(test_data, pred_data,itemlabel):
    r = []
    entropy_ratio = []
    entropy_pred = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        groundtruelabel = itemlabel[np.array(groundTrue).squeeze()]
        groundTrue_entropy = entropy1(groundtruelabel)
        predictTopK = pred_data[i]
        predictlabel = itemlabel[predictTopK]
        predictTopK_entropy = entropy1(predictlabel)
        enratio = predictTopK_entropy / (groundTrue_entropy + np.power(0.1,5))
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
        if enratio < 100:
            entropy_ratio.append(enratio)
            entropy_pred.append(predictTopK_entropy)
    return np.array(r).astype('float'), np.array(entropy_ratio), np.array(entropy_pred)

# ====================end Metrics=============================
# =========================================================
