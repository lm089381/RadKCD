import torch
import torch.nn as nn
import numpy as np
import LIMITR_loss

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class CELoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=0.1)

    def forward(self, output, target):
        '''
        Output: (N,*,C) \n
        Target: (N,*) \n
        '''
        output = torch.log(output)
        output = output.reshape(-1, output.shape[-1])
        target = target.reshape(-1).long()
        return self.CELoss(output, target)

class CELossShift(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss(ignore_index=ignore_index)

    def forward(self, output, target):
        output = output[:,:-1,:]
        target = target[:,1:]
        return self.CELoss(output, target)

class CELossTotal(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss()
        self.CELossShift = CELossShift(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1])

class LIMITR(nn.Module):
    def __init__(self):
        super(LIMITR, self).__init__()

        self.local_ext_loss = LIMITR_loss.local_ext_loss
        self.global_loss = LIMITR_loss.global_loss
        self.local_int_loss = LIMITR_loss.local_int_loss

        self.local_ext_loss_weight = 1.0
        self.global_loss_weight = 1.0

        self.temp1 = 4.0
        self.temp2 = 5.0
        self.temp3 = 10.0

        self.SAF_module = SA(1000, 0.4)
        self.sim_eval_w_cap = nn.Linear(1000, 1)
        self.sim_eval_w_img = nn.Linear(1000, 1)

        self.sigmoid = nn.Sigmoid()
        self.l2norm = l2norm
        self.conv = nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1)
        # self.global_embedder = nn.Linear(1024, 256)

    def attention_fn(self,query, context, temp1,context_img):
        """
        query: batch x ndf x queryL
        context: batch x ndf x ih x iw (sourceL=ihxiw)
        mask: batch_size x sourceL
        """
        #当context_img=True时，attention_fn 表示的是 “文本（Query）对图像区域（Context）的注意力”
        if context_img:
            batch_size, queryL = query.size(0), query.size(2)
            sourceL = context.size(2)
            contextT = torch.transpose(context, 1, 2).contiguous()
        else:
            batch_size = query.size(0)
            sourceL = context.size(1)
            queryL = query.size(2)
            contextT = context

        # Get attention
        # (batch x sourceL x ndf)(batch x ndf x queryL)
        # -->batch x sourceL x queryL
        print(contextT.shape)   #[8, 128, 1000]
        print(query.shape)  #[8, 114, 256]
        attn = torch.bmm(contextT, query)
        # --> batch*sourceL x queryL
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax(dim=-1)(attn) #这里得到的是每个图像区域对各单词的注意力，并确保每个图像区域对所有单词的注意力权重之和为1（横向归一化）

        # --> batch x sourceL x queryL
        attn = attn.view(batch_size, sourceL, queryL)
        # --> batch*queryL x sourceL
        attn = torch.transpose(attn, 1, 2).contiguous()
        attn = attn.view(batch_size * queryL, sourceL)  #这里得到的是每个单词对各图像区域的注意力

        attn = attn * temp1
        attn = nn.Softmax(dim=-1)(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # (batch x ndf x sourceL)(batch x sourceL x queryL)
        # --> batch x ndf x queryL
        if context_img:
            weightedContext = torch.bmm(context, attnT)
        else:
            contextT = torch.transpose(context, 1, 2).contiguous()
            weightedContext = torch.bmm(contextT, attnT)

        return weightedContext, attn


    def local_similarities(
            self, img_features_l, img_features_f, words_emb, temp1
    ):
        img_features_l = self.conv(img_features_l)  #[8,1000,8,8]
        img_features_f = self.conv(img_features_f)  #[8,1000,8,8]
        batch_size = img_features_f.shape[0]    #8

        img_features_f = img_features_f.type(words_emb.dtype)
        img_features_l = img_features_l.type(words_emb.dtype)

        ih, iw = img_features_f.size(2), img_features_f.size(3)
        sourceL = ih * iw

        img_features_f = img_features_f.view(batch_size, -1, sourceL)   #[8,1000,64]
        img_features_l = img_features_l.view(batch_size, -1, sourceL)   #[8,1000,64]

        # concat frontal + lateral features
        img_features = torch.cat((img_features_f, img_features_l), dim=2)   #[8,1000,128]

        similarities_cap = []
        similarities_img = []
        att_maps = []

        for i in range(words_emb.shape[0]):
            word = words_emb[i].unsqueeze(0).contiguous()
            word = word.repeat(batch_size, 1, 1)
            context = img_features

            weiContext, attn = self.attention_fn(
                word, context, temp1, context_img=True
            )

            att_maps.append(attn[i].unsqueeze(0).contiguous())

            word = word.transpose(1, 2).contiguous()
            weiContext = weiContext.transpose(1, 2).contiguous()

            sim_loc_cap = torch.mul(weiContext, word)
            sim_loc_cap = self.l2norm(sim_loc_cap, dim=-1)
            sim_ave_cap = torch.mean(sim_loc_cap, 1)

            # print(sim_loc_cap.shape)    #torch.Size([8, 256, 1000])
            # print(sim_ave_cap.shape)    #torch.Size([8, 1000])
            sim_vec_cap, sim_w_cap = self.SAF_module(sim_loc_cap, sim_ave_cap)
            sim_i_cap = self.sigmoid(self.sim_eval_w_cap(sim_vec_cap))
            similarities_cap.append(sim_i_cap)

        similarities_cap = torch.cat(similarities_cap, 1)

        return similarities_cap, att_maps


    def _calc_local_loss(self, img_emb_l_l,img_emb_l_f, text_emb_l, pad_mask):

        # sim_loc_cap, attn_maps = self.local_similarities(img_emb_l_l,
        #     img_emb_l_f,
        #     text_emb_l,
        #     temp1=self.temp1)
        # l_loss0_cap, l_loss1_cap,l_loss0_img, l_loss1_img = self.local_ext_loss(
        #     sim_loc_cap,
        #     temp3=self.temp3
        # )
        # sim_loc = sim_loc_cap.cuda()
        # l_loss0 = 0.5 * (l_loss0_cap+l_loss0_img)
        # l_loss1 = 0.5 * (l_loss1_cap + l_loss1_img)

        local_int_loss = self.local_int_loss(img_emb_l_l, img_emb_l_f, text_emb_l, self.SAF_module, pad_mask, local_temperature=0.1)

        return local_int_loss
        # return l_loss0, l_loss1, local_int_loss, sim_loc, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        #img_emb_g的形状是[8,114,256],text_emb_g的形状是[8,114,256]
        # img_emb_g = self.global_embedder(img_emb_g)
        g_loss0, g_loss1,sim_glo = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1,sim_glo

    def calc_loss(self, img_emb_l_l, img_emb_l_f, img_emb_g, text_emb_l, text_emb_g,pad_mask):
        loss = 0

        # l_loss0, l_loss1, local_int_loss, sim_loc, attn_maps = self._calc_local_loss(
        #     img_emb_l_l, img_emb_l_f, text_emb_l, pad_mask)
        local_int_loss = self._calc_local_loss(
            img_emb_l_l, img_emb_l_f, text_emb_l, pad_mask)
        # local_ext_loss = (l_loss0 + l_loss1)
        # loss += local_ext_loss * 0.5
        loss += local_int_loss * 0.5

        g_loss0, g_loss1, sim_glo = self._calc_global_loss(img_emb_g, text_emb_g)
        global_loss = (g_loss0 + g_loss1)
        loss += global_loss * 1.0

        # similarities = torch.stack(
        #     [0.5*sim_loc, 1.0*sim_glo]
        # )

        # similarities = similarities.mean(axis=0)
        # weighted loss
        # return loss, local_ext_loss, local_int_loss, global_loss, similarities, sim_loc, sim_glo, attn_maps
        return loss, local_int_loss, global_loss

    def forward(self, x):
        # img encoder branch
        img_emb_l_l, img_emb_l_f, img_emb_g, ind_lateral = self.image_encoder_forward(x["imgs_F"], x["imgs_L"])
        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        return img_emb_l_l, img_emb_l_f, ind_lateral, img_emb_g, text_emb_l, text_emb_g, sents

class SA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(SA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)
        self.l2norm = l2norm

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = self.l2norm(new_global, dim=-1)

        return new_global, weights


