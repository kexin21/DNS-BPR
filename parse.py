import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go DNS-BPR")
    parser.add_argument('--batch_size', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--evaluation_step', type=int,default=10,
                        help="evaluation_step")
    parser.add_argument("--ns", type=str, default='dns', help="rns,dns")
    parser.add_argument("--n_negs", type=int, default=100, help="number of candidate negative")
    parser.add_argument("--K", type=int, default=1, help="number of negatives for each positive user-item pair")

    parser.add_argument("--loss", type=str, default='abc_bpr', help="bpr,abc_bpr")
    parser.add_argument("--alpha_bpr", type=float, default=0.5, help="a value if use ab_bpr, mainly control y-axis direction")
    parser.add_argument("--beta_bpr", type=float, default=-1, help="b value if use ab_bpr, mainly control x-axis direction")
    parser.add_argument("--c_bpr", type=float, default=-1, help="c value if use ab_bpr, mainly control x-axis direction")


    parser.add_argument("--cuda", type=bool, default=False, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")
    parser.add_argument("--context_hops", type=int, default=3, help="hop")

    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--test_batch_size', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='taobao',
                        help="available datasets: [taobao,Tmall,gowalla]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20,50]",
                        help="@k test list")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=600)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--model', type=str, default='mf', help='rec-model, support [mf, lgn]')


    return parser.parse_args()
