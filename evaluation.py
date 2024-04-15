import numpy as np
import torch
import utils
import multiprocessing
from parse import parse_args
import matplotlib.pyplot as plt


CORES = multiprocessing.cpu_count() // 2
args = parse_args()
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
Ks = eval(args.topks)
multicore = args.multicore
plt.rcParams.update({'font.size':23})

def test_one_batch_with_entropy(X, itemlabel):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, entropy, coverage = [], [], [], [], []
    for k in Ks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        entropy.append(utils.ENTROPY_ATk(sorted_items, k, itemlabel))
        coverage.append(utils.COVERAGE(sorted_items, k, itemlabel))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg),
            'entropy':np.array(entropy),
            'coverage':np.array(coverage)}

def Test_with_entropy(dataset, Recmodel, epoch, w=None, mode='test'):
    max_K = max(Ks)

    if mode == 'test':
        testDict = dataset.test_user_set
    elif mode == 'val':
        testDict = dataset.val_user_set
    elif mode == 'train':
        testDict = dataset.train_user_set

    pool = multiprocessing.Pool(CORES)

    results = {'precision': np.zeros(len(Ks)),
               'recall': np.zeros(len(Ks)),
               'ndcg': np.zeros(len(Ks)),
               'entropy': np.zeros(len(Ks)),
               'coverage': np.zeros(len(Ks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert BATCH_SIZE <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []

        total_batch = len(users) // BATCH_SIZE + 1
        # total_batch = len(users[:100]) // u_batch_size
        user_emb, item_emb = Recmodel.generate()

        for batch_users in utils.minibatch(users, batch_size=BATCH_SIZE):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)
            if args.model == 'mf':
              u_embeddings = user_emb(batch_users_gpu)
            else:
              u_embeddings = user_emb[batch_users_gpu]

            item_batch = torch.LongTensor(np.array(range(0, dataset.m_items))).view(dataset.m_items, -1).squeeze(dim=1).to(device)
            if args.model == 'mf':
              i_embeddings = item_emb(item_batch)
            else:
              i_embeddings = item_emb[item_batch]
            rating = Recmodel.rating(u_embeddings, i_embeddings).detach().cpu()

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            if mode == 'train':
              pass
            else:
              rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        # assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch_with_entropy, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch_with_entropy(x, dataset.category_label))
        scale = float(BATCH_SIZE/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['entropy'] += result['entropy']
            results['coverage'] += result['coverage']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['entropy'] /= float(len(users))
        results['coverage'] /= float(len(users))
        if w is not None:
            w.add_scalar('Recall@50',results['recall'][1],epoch)
            w.add_scalar('Recall@20',results['recall'][0],epoch)
            w.add_scalar('NDCG@50',results['ndcg'][1],epoch)
            w.add_scalar('NDCG@20',results['ndcg'][0],epoch)
        if multicore == 1:
            pool.close()
        print(results)
        return results





