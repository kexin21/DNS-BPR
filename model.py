import torch
import numpy as np
from torch import nn

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
    
class PureMF(BasicModel):
    def __init__(self, args, dataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        # self.noise = dataset.noise
        self.latent_dim = args.recdim
        self.ns = args.ns
        self.K = args.K
        self.n_negs = args.n_negs
        self.decay = args.decay
        self.f = nn.Sigmoid()
        self.__init_weight()

        self.loss = args.loss

        self.a = args.alpha_bpr
        self.b = args.beta_bpr
        self.c = args.c_bpr
        self.device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def generate(self):
        return self.embedding_user, self.embedding_item

    def rating(self, u_emb=None, i_emb=None):
        return torch.matmul(u_emb, i_emb.t())

    def bpr_loss(self, user_emb, pos_embs, neg_embs):
        batch_size = user_emb.shape[0]

        u_e = user_emb
        pos_e = pos_embs
        neg_e = neg_embs

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)

        pos_neg = pos_scores.unsqueeze(dim=1) - neg_scores

        if self.loss == 'abc_bpr':
            mf_loss = torch.mean(-torch.log((self.a + 1 / (1 + torch.exp(
                self.b + (neg_scores - pos_scores.unsqueeze(dim=1)) * self.c))) / (
                                                        1 + self.a)))
        elif self.loss == 'bpr':
            mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1))))

        regularize = (torch.norm(user_emb[:, :]) ** 2
                       + torch.norm(pos_embs[:, :]) ** 2
                       + torch.norm(neg_embs[:, :, :]) ** 2 / self.K) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss, pos_neg.detach().cpu().numpy()

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        if self.ns == 'rns':
            neg_embs = self.embedding_item(neg_item[:,:self.K])
        elif self.ns == 'dns':
            neg_embs = []
            for k in range(self.K):
                neg_embs.append(self.negative_sampling_dns(self.embedding_user, self.embedding_item,
                                                           user, neg_item[:, k*self.n_negs: (k+1)*self.n_negs]))
            neg_embs = torch.stack(neg_embs, dim=1).squeeze(2)
        else:
            pass
        return self.bpr_loss(self.embedding_user(user), self.embedding_item(pos_item), neg_embs)

    def negative_sampling_dns(self, user_emb, item_emb, user, neg_candidates):
        batch_size = user.shape[0]
        s_e = user_emb(user)
        n_e = item_emb(neg_candidates)

        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)
        indices = torch.max(scores, dim=1)[1].detach()
        indices = indices.unsqueeze(1).unsqueeze(2)

        return torch.take_along_dim(n_e,indices,dim=1)
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]

class LightGCN(BasicModel):
    def __init__(self, args, dataset):
        super(LightGCN, self).__init__()
        self.n_users  = dataset.n_users
        self.n_items  = dataset.m_items
        self.sparse_norm_adj = dataset.getSparseGraph()
        self.emb_size = args.recdim
        self.context_hops = args.context_hops
        self.mess_dropout = args.mess_dropout
        self.mess_dropout_rate = args.mess_dropout_rate
        self.edge_dropout = args.edge_dropout
        self.edge_dropout_rate = args.edge_dropout_rate
        self.pool = args.pool
        self.ns = args.ns
        self.K = args.K

        self.loss = args.loss

        self.n_negs = args.n_negs
        self.decay = args.decay
        self.device = torch.device("cuda:0") if args.cuda else torch.device("cpu")

        self.a = args.alpha_bpr
        self.b = args.beta_bpr
        self.c = args.c_bpr
        # self.c = 1
        self.mean = 0

        self.epoch = 0

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        
        self.gcn = self._init_model()


    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        self.Norm = torch.distributions.Normal(-0.2,1.4)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
            neg_labels = None
        elif self.ns == 'dns':
            neg_gcn_embs = []
            neg_labels = []
            for k in range(self.K):
                emb, label = self.negative_sampling_dns_original(user_gcn_emb, item_gcn_emb,
                                                           user, neg_item[:, k*self.n_negs: (k+1)*self.n_negs],
                                                           pos_item)
                neg_gcn_embs.append(emb)
                neg_labels.append(label)
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)
            neg_labels = torch.stack(neg_labels, dim=1).squeeze(-1).squeeze(-1)
        return self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def negative_sampling_dns_original(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel] [2048,4,64]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1) # [2048,1,64]

        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel] [2048,8,4,64]
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel] [2048,4,8,64]

        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1) # [batch_size, n_negs, n_hops+1]
        scores = scores.sum(dim=-1) # [2048,8]
        indices = torch.max(scores, dim=1)[1].unsqueeze(1) # [2048,1]
        indices_repeat = indices.unsqueeze(dim=-1).unsqueeze(dim=-1).detach() # [2048,1,1,1]

        neg_label = torch.take_along_dim(neg_candidates, indices, dim=-1)

        neg_emb = torch.take_along_dim(neg_items_emb_, indices_repeat, dim=2).squeeze(dim=-2)
        return neg_emb, neg_label


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))


    def \
            pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size,
                                                                                                       self.K, -1)
        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]

        if self.loss == 'abc_bpr':
            mf_loss = torch.mean(-torch.log((self.a + 1 / (1 + torch.exp(
                self.b + (neg_scores - pos_scores.unsqueeze(dim=1)) * self.c))) / (
                                                        1 + self.a)))
        elif self.loss == 'bpr':
            mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1))))

        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2 / self.K) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss





