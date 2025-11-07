import torch
import torch.nn as nn
from dam import DAM
from model.vlci import VLCI
import torch.nn.functional as F
from gm_generator import GuidanceMemoryGenerator, ContextGuidanceNorm

class GateFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GateFusion, self).__init__()
        self.gate_fc = nn.Linear(512, 1)

    def forward(self, A, B):
        # A, B: [B, C, H, W]
        # print(A.shape)    #torch.Size([8, 114, 256])
        # print(B.shape)    #torch.Size([8, 1000, 256])
        B_proj = F.interpolate(B.permute(0, 2, 1), size=114, mode='linear', align_corners=False).permute(0, 2, 1)

        fusion = torch.cat([A, B_proj], dim=-1)  # [B, 2C, H, W]
        gate = torch.sigmoid(self.gate_fc(fusion))  # [B, 1, H, W]，值在[0,1]

        output = gate * A + (1 - gate) * B_proj
        return output


def extractor_patch_features(feature_map, patch_size=1):
    B, C, H, W = feature_map.shape

    if patch_size == 1:
        # patch_features = feature_map.permute(0, 2, 3, 1).reshape(B, -1, C)   #(B, H*W, C)
        patch_features = feature_map.reshape(B, C, -1)  # (B, H*W, C)
    else:
        patches = feature_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size,
                                            patch_size)  # (B, C, num_patches, patch_size, patch_size)
        patch_features = patches.mean(dim=[-1, -2])  # 对每个patch提取特征，在最后两个维度上取均值
        patch_features = patch_features.permute(0, 2, 1)  # (B, num_patches, C)

    return patch_features

def attention_fn(query, context, temp1=5.0, context_img=True):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
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
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

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

class SelfAttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        # 定义 Q, K, V 的线性变换
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x 形状: [B, V, D] (对avg) 或 [B, V, C', W'*H'] (对wxh展平后)
        B, V, _ = x.shape
        Q = self.query(x)  # [B, V, D]
        K = self.key(x)  # [B, V, D]
        V = self.value(x)  # [B, V, D]

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # [B, V, V]
        attn_weights = F.softmax(scores, dim=-1)  # 按行归一化

        # 加权求和
        fused = torch.matmul(attn_weights, V)  # [B, V, D]
        return fused.mean(dim=1)  # 聚合所有视角 [B, D]

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        input = input.permute(1, 0, 2)
        query = query.permute(1, 0, 2)
        embed, att = self.attention(query, input, input, key_padding_mask=pad_mask, attn_mask=att_mask)

        embed = self.normalize(embed + query)
        embed = embed.permute(1, 0, 2)
        return embed, att

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention_image_to_text = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.attention_text_to_image = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, img_features, text_features, cross_embed, pad_mask=None, att_mask=None):
        img_features = img_features.permute(1, 0, 2)
        text_features = text_features.permute(1, 0, 2)
        cross_embed = cross_embed.permute(1, 0, 2)
        eimg_features = img_features + cross_embed

        refined_text, att_1 = self.attention_image_to_text(eimg_features, text_features, text_features,
                                                           key_padding_mask=pad_mask,
                                                           attn_mask=att_mask)  #[114,64,256]

        refined_img, att_2 = self.attention_text_to_image(eimg_features, refined_text, eimg_features,
                                                          key_padding_mask=pad_mask,
                                                          attn_mask=att_mask)

        refined_img = refined_img.permute(1, 0, 2)  #[64,114,256]
        refined_text = refined_text.permute(1, 0, 2)
        return refined_img, refined_text, att_1, att_2

class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, fwd_dim, dropout=0.0):
        super().__init__()
        self.fwd_layer = nn.Sequential(
            nn.Linear(emb_dim, fwd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fwd_dim, emb_dim),
        )
        self.normalize = nn.LayerNorm(emb_dim)

    def forward(self, input):
        output = self.fwd_layer(input)
        output = self.normalize(output + input)
        return output


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, pad_mask=None, att_mask=None):
        emb, att = self.attention(input, input, pad_mask, att_mask)
        emb = self.fwd_layer(emb)
        return emb, att


class GenTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0, gm_mem_len=3):
        super().__init__()
        self.self_attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.cross_attention = MultiheadAttention(embed_dim, num_heads, dropout)

        "------------------"
        # self.cnl = ContextGuidanceNorm(embed_dim, gm_mem_len * embed_dim)
        self.gate = GateFusion(embed_dim)
        self.normal = nn.LayerNorm(embed_dim)
        # self.img_features = nn.Linear(1024, 256)

        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input,hv,ht, context=None, pad_mask=None, cross_pad_mask=None, att_mask=None, gm=None):
        hidden_states, self_att = self.self_attention(input, input, pad_mask, att_mask)
        "-----------------------"
        # if gm is not None:
        #     hidden_states = self.cnl(hidden_states, gm)
        # print(hv.shape)
        # print(ht.shape)   #torch.Size([8, 1000, 256])
        # print(hidden_states.shape)    #torch.Size([8, 1114, 256])
        # print(context.shape)  #torch.Size([8, 114, 256])
        f_gate = self.gate(context, ht)
        # f_fuse = self.normal(context+ht)


        if context is not None:
            enhanced_states, cross_att = self.cross_attention(
                f_gate,
                hidden_states,
                cross_pad_mask
            )

            # hidden_states = enhanced_states     #torch.Size([8, 1114, 256])
            hidden_states = enhanced_states     #torch.Size([8, 1114, 256])

        "----------------------"
        # if gm is not None:
        #     hidden_states = self.cnl(hidden_states, gm)

        output = self.fwd_layer(hidden_states)
        return output, self_att


class TNN(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.1, num_layers=1,
                 num_tokens=1, num_posits=1, token_embedding=None, posit_embedding=None):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim) if not token_embedding else token_embedding
        self.posit_embedding = nn.Embedding(num_posits, embed_dim) if not posit_embedding else posit_embedding
        self.transform = nn.ModuleList(
            [TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)


    def forward(self, token_index=None, token_embed=None, pad_mask=None, pad_id=-1, att_mask=None):
        if token_index != None:
            if pad_mask == None:
                pad_mask = (token_index == pad_id)
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)
            posit_embed = self.posit_embedding(posit_index)
            token_embed = self.token_embedding(token_index)
            final_embed = self.dropout(token_embed + posit_embed)
        elif token_embed != None:
            posit_index = torch.arange(token_embed.shape[1]).unsqueeze(0).repeat(token_embed.shape[0], 1).to(
                token_embed.device)
            posit_embed = self.posit_embedding(posit_index)
            final_embed = self.dropout(token_embed + posit_embed)
        else:
            raise ValueError('token_index or token_embed must not be None')

        for i in range(len(self.transform)):
            final_embed = self.transform[i](final_embed, pad_mask, att_mask)[0]

        return final_embed

class CNN(nn.Module):
    def __init__(self, model, model_type='resnet'):
        super().__init__()
        self.extractor_patch_features = extractor_patch_features
        self.dam = DAM(1024, 1024)
        if 'res' in model_type.lower():
            self.feature = nn.Sequential(*list(model.children())[:-2])
            self.average = list(model.children())[:-1][-1]
        elif 'dense' in model_type.lower():
            modules = list(model.features.children())[:-1]
            self.feature = nn.Sequential(*modules)
            self.average = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Unsupported model_type!')

    def forward(self, input):
        # print(input.shape)    #[16,3,256,256]
        orig_features = self.feature(input)
        # print(orig_features.shape)  #torch.Size([16, 1024, 8, 8])
        patch_features = self.extractor_patch_features(orig_features, 1)
        # print(patch_features.shape) #torch.Size([16, 64, 1024])
        B = patch_features.shape[0]
        # patch_features = patch_features.view(-1, patch_features.shape[-1])
        feat_sum = self.dam(orig_features)
        # print(feat_sum.shape)   #torch.Size([16, 1024, 8, 8])
        # feat_repeated = feat_sum.unsqueeze(0).repeat(B, 1, 1)
        # avg_features = self.average(orig_features)
        avg_features = self.average(feat_sum)
        avg_features = avg_features.view(avg_features.shape[0], -1) #[16,1024]

        return avg_features, feat_sum   #[16,1024]|[16, 1024, 8, 8]，全局和局部
        # return avg_features, orig_features

class MVCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  #这里的model是上面的CNN实例
        self.attn_fusion =  SelfAttentionFusion(feature_dim=1024)

    def forward(self, input):
        img = input[0]  #表示 B 个物体，每个物体有 V 张视角图片（比如从不同角度拍摄的图片），每张图片的尺寸是 C 通道 × W 宽 × H 高
        pos = input[1]  #pos 是一个形状为 [B, V] 的张量，如果 pos[b, v] == -1，表示第 b 个物体的第 v 个视角是无效的
        B, V, C, W, H = img.shape   #8,2,3,256,256

        img = img.view(B * V, C, W, H)
        avg, wxh = self.model(img)  #img.shape:[16,3,256,256],所以单张图片的形状是[8,3,256,256]   #[16,1024]|[16, 1024, 8, 8]
        avg = avg.view(B, V, -1)
        wxh = wxh.view(B, V, wxh.shape[-3], wxh.shape[-2], wxh.shape[-1])

        msk = (pos == -1)
        msk_wxh = msk.view(B, V, 1, 1, 1).float()
        msk_avg = msk.view(B, V, 1).float()
        wxh = msk_wxh * (-1) + (1 - msk_wxh) * wxh  #把无效视角的特征替换成 -1，保留有效视角的特征
        avg = msk_avg * (-1) + (1 - msk_avg) * avg

        # 保留侧视图（视角 0）和正视图（视角 1）的局部特征
        wxh_l = wxh[:, 0]  # [B, 1024, 8, 8]
        wxh_f = wxh[:, 1]  # [B, 1024, 8, 8]

        #融合多视角特征
        wxh_features = wxh.max(dim=1)[0]
        # avg_features = avg.max(dim=1)[0]
        avg_features = self.attn_fusion(avg)  # [B, D]
        return avg_features, wxh_features, wxh_l, wxh_f   #[8,1024]|[8,1024,8,8]，全局和局部


# --- Main Moduldes ---
class Classifier(nn.Module):
    def __init__(self, num_topics, num_states, cnn=None, tnn=None,
                 fc_features=2048, embed_dim=128, num_heads=1, dropout=0.1):
        super().__init__()
        self.cnn = cnn
        self.tnn = tnn
        self.img_features = nn.Linear(fc_features, num_topics * embed_dim) if cnn != None else None     #nn.Linear(1024,114*256)
        self.txt_features = MultiheadAttention(embed_dim, num_heads, dropout) if tnn != None else None
        self.cross_features = CrossModalAttention(embed_dim, num_heads, dropout)
        self.topic_embedding = nn.Embedding(num_topics, embed_dim)  #num_topics:14+100=114
        self.state_embedding = nn.Embedding(num_states, embed_dim)  #num_states:2
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.num_topics = num_topics
        self.num_states = num_states
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(embed_dim)
        self.projection = nn.Linear(1114, 114)
        '------------------------'
        self.vlci = VLCI()
        self.extractor_patch_features = extractor_patch_features
        self.proj1 = nn.Linear(1024, 256)
        self.pool1 = nn.AdaptiveAvgPool1d(114)

    def forward(self, img=None, txt=None, lbl=None, txt_embed=None, pad_mask=None, pad_id=3, threshold=0.5,
                get_embed=False, get_txt_att=False):
        if img != None:
            img_features, wxh_features, wxh_l, wxh_f = self.cnn(img)    #[8,1024]|[8,1024,8,8]，全局和局部
            img_features = self.dropout(img_features)
            wxh_features = self.extractor_patch_features(wxh_features, 1).transpose(1, 2)   #[8,64,1024]

        if txt != None:
            if pad_id >= 0 and pad_mask == None:
                pad_mask = (txt == pad_id)
            txt_features = self.tnn(token_index=txt, pad_mask=pad_mask)

        elif txt_embed != None:
            txt_features = self.tnn(token_embed=txt_embed, pad_mask=pad_mask)

        hv, ht = self.vlci(wxh_features, txt_features, mode='train')

        if img != None and (txt != None or txt_embed != None):
            #生成一个形状为 (batch_size, self.num_topics) 的整数索引张量
            cross_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)
            memory_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0], 1).to(
                img_features.device)
            cross_embed = self.topic_embedding(cross_index)
            memory_embed = self.state_embedding(memory_index)
            #这行代码对应论文中的：将转换后的特征向量映射成n×e维，然后被重塑为矩阵I
            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics, -1)
            # hv = self.pool1(self.proj1(hv).transpose(1, 2)).transpose(1, 2)
            # img_features = self.normalize(img_features + hv)
            # txt_features = self.normalize(txt_features + ht)

            img_features, txt_features, att_1, att_2 = self.cross_features(img_features, txt_features, cross_embed)   #torch.Size([8, 114, 256])
        #     final_embed = self.normalize(img_features + txt_features)   #融合特征Z
        #
        # else:
        #     raise ValueError('img and txt error')
        #
        # mem_emb, att = self.attention(memory_embed, final_embed)

            # 获取医学知识嵌入
            mem_emb, att = self.attention(memory_embed, img_features + txt_features)
            if lbl != None:
                emb = self.state_embedding(lbl)
            else:
                emb = self.state_embedding((att[:, :, 1] > threshold).long())

            # 将医学知识与图像特征和文本特征分别结合
            img_features_with_mem = img_features + emb
            txt_features_with_mem = txt_features + emb

            # 融合结合后的特征
            final_embed = self.normalize(img_features_with_mem + txt_features_with_mem)

        else:
            raise ValueError('img and txt error')

        # if lbl != None:
        #     emb = self.state_embedding(lbl)     #训练阶段，如果标签已知，直接用真实标签的embedding训练下游任务
        # else:
        #     emb = self.state_embedding((att[:, :, 1] > threshold).long())   #推理阶段，没有标签了，只能依靠attention输出判断哪些知识状态应该激活，然后将这些激活的状态向量作为医学知识表示
        # emb2 = self.state_embedding((att[:, :, 1] > threshold).long())

        if get_embed:
            # return att, final_embed + emb, final_embed + emb2
            return att, final_embed, final_embed,hv,ht
        else:
            return att

class Generator(nn.Module):
    def __init__(self, num_tokens, num_posits, embed_dim=128, num_heads=1, fwd_dim=256, dropout=0.1, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.posit_embedding = nn.Embedding(num_posits, embed_dim)
        self.transform = nn.ModuleList(
            [GenTransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        # 单独的多头注意力层，用于最后的 token-to-context 注意力提取
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.num_tokens = num_tokens
        self.num_posits = num_posits
        "---------------"
        # self.fuse_layer = nn.Linear(embed_dim, embed_dim)
        # self.activation = nn.ReLU()
        # self.gm_module = GuidanceMemoryGenerator(embed_dim, embed_dim, num_heads)

    # def forward(self, source_embed, source_embed2, token_index=None, source_pad_mask=None, target_pad_mask=None,
    #             max_len=300, top_k=1, bos_id=1, pad_id=3, mode='eye', gm=None):
    #     from torch.utils.checkpoint import checkpoint
    #
    #     if gm is None:
    #         gm = torch.zeros(source_embed.shape[0], 3, source_embed.shape[2]).to(source_embed.device)
    #
    #     if token_index is not None:
    #         B, T = token_index.shape
    #         device = token_index.device
    #
    #         posit_index = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
    #         position_embeddings = self.posit_embedding(posit_index)
    #         token_embeddings = self.token_embedding(token_index)
    #
    #         if source_pad_mask is None:
    #             source_pad_mask = torch.zeros((B, source_embed.shape[1]), device=device).bool()
    #
    #         outputs = []
    #         for t in range(T):
    #             # 每次输入一个token
    #             token_input = token_embeddings[:, t:t+1, :]
    #             pos_input = position_embeddings[:, t:t+1, :]
    #             target_input = token_input + pos_input
    #
    #             final_embed = torch.cat([source_embed, target_input], dim=1)
    #
    #             target_pad_t = target_pad_mask[:, t:t+1] if target_pad_mask is not None else torch.zeros((B, 1), device=device).bool()
    #             pad_mask = torch.cat([source_pad_mask, target_pad_t], dim=1)
    #             att_mask = self.generate_square_subsequent_mask_with_source(source_embed.shape[1], 1, mode).to(device)
    #
    #             gm = self.gm_module(gm, target_input)
    #
    #             for i in range(len(self.transform)):
    #                 def custom_forward(x):
    #                     return self.transform[i](x, source_embed2, pad_mask, source_pad_mask, att_mask, gm)[0]
    #                 final_embed = checkpoint(custom_forward, final_embed)
    #
    #             outputs.append(final_embed[:, -1:, :].detach())
    #
    #         output_seq = torch.cat(outputs, dim=1)
    #
    #         token_index_full = torch.arange(self.num_tokens, device=device).unsqueeze(0).expand(B, -1)
    #         token_embed = self.token_embedding(token_index_full)
    #         emb, att = self.attention(token_embed, torch.cat([source_embed, output_seq], dim=1))
    #         emb = emb[:, source_embed.shape[1]:, :]
    #         att = att[:, source_embed.shape[1]:, :]
    #         return att, emb, gm
    #     else:
    #         return self.infer(source_embed, source_embed2, source_pad_mask, max_len, top_k, bos_id, pad_id)
    #
    # def infer(self, source_embed, source_embed2, source_pad_mask=None, max_len=100, top_k=1, bos_id=1, pad_id=3):
    #     outputs = torch.ones((top_k, source_embed.shape[0], 1), dtype=torch.long).to(source_embed.device) * bos_id
    #     scores = torch.zeros((top_k, source_embed.shape[0]), dtype=torch.float32).to(source_embed.device)
    #     gms = [torch.zeros(source_embed.shape[0], 3, source_embed.shape[2], device=source_embed.device) for _ in range(top_k)]
    #
    #     for _ in range(1, max_len):
    #         possible_outputs = []
    #         possible_scores = []
    #         possible_gms = []
    #
    #         for k in range(top_k):
    #             output = outputs[k]
    #             score = scores[k]
    #             gm = gms[k]
    #
    #             att, emb, gm_updated = self.forward(
    #                 source_embed, source_embed2, output,
    #                 source_pad_mask=source_pad_mask,
    #                 target_pad_mask=(output == pad_id),
    #                 gm=gm
    #             )
    #
    #             val, idx = torch.topk(att[:, -1, :], top_k)
    #             log_val = -torch.log(val)
    #
    #             for i in range(top_k):
    #                 new_output = torch.cat([output, idx[:, i].view(-1, 1)], dim=-1)
    #                 new_score = score + log_val[:, i].view(-1)
    #                 possible_outputs.append(new_output.unsqueeze(0))
    #                 possible_scores.append(new_score.unsqueeze(0))
    #                 possible_gms.append(gm_updated.unsqueeze(0))
    #
    #         possible_outputs = torch.cat(possible_outputs, dim=0)
    #         possible_scores = torch.cat(possible_scores, dim=0)
    #         possible_gms = torch.cat(possible_gms, dim=0)
    #
    #         val, idx = torch.topk(possible_scores, top_k, dim=0)
    #         col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)
    #
    #         outputs = possible_outputs[idx, col_idx]
    #         scores = possible_scores[idx, col_idx]
    #         gms = possible_gms[idx, col_idx]
    #
    #     val, idx = torch.topk(scores, 1, dim=0)
    #     col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)
    #     output = outputs[idx, col_idx]
    #     return output.squeeze(0)

    def forward(self, source_embed, source_embed2,hv,ht, token_index=None, source_pad_mask=None, target_pad_mask=None,
                max_len=300, top_k=1, bos_id=1, pad_id=3, mode='eye'):

        "-----------------"
        # gm = torch.zeros(source_embed.shape[0], 1, source_embed.shape[2]).to(source_embed.device)

        if token_index != None:
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)
            posit_embed = self.posit_embedding(posit_index)
            token_embed = self.token_embedding(token_index)
            target_embed = token_embed + posit_embed

            final_embed = torch.cat([source_embed, target_embed], dim=1)

            "-----------------------"
            # final_embed = self.activation(self.fuse_layer(final_embed))

            if source_pad_mask == None:
                source_pad_mask = torch.zeros((source_embed.shape[0], source_embed.shape[1]),
                                              device=source_embed.device).bool()
            if target_pad_mask == None:
                target_pad_mask = torch.zeros((target_embed.shape[0], target_embed.shape[1]),
                                              device=target_embed.device).bool()
            pad_mask = torch.cat([source_pad_mask, target_pad_mask], dim=1)
            cross_pad_mask = source_pad_mask
            att_mask = self.generate_square_subsequent_mask_with_source(source_embed.shape[1], target_embed.shape[1],
                                                                        mode).to(final_embed.device)

            for i in range(len(self.transform)):
                # gm = self.gm_module(gm, final_embed[:, -1:, :])  # 更新 GM
                final_embed = self.transform[i](final_embed,hv,ht, source_embed2, pad_mask, cross_pad_mask, att_mask)[0]
            token_index = torch.arange(self.num_tokens).unsqueeze(0).repeat(token_index.shape[0], 1).to(
                token_index.device)
            token_embed = self.token_embedding(token_index)
            emb, att = self.attention(token_embed, final_embed)
            emb = emb[:, source_embed.shape[1]:, :]
            att = att[:, source_embed.shape[1]:, :]
            return att, emb
        else:
            return self.infer(source_embed, source_embed2,hv,ht, source_pad_mask, max_len, top_k, bos_id, pad_id)

    def infer(self, source_embed, source_embed2,hv,ht, source_pad_mask=None, max_len=100, top_k=1, bos_id=1, pad_id=3):
        outputs = torch.ones((top_k, source_embed.shape[0], 1), dtype=torch.long).to(source_embed.device) * bos_id
        scores = torch.zeros((top_k, source_embed.shape[0]), dtype=torch.float32).to(source_embed.device)
        for _ in range(1, max_len):
            possible_outputs = []
            possible_scores = []
            for k in range(top_k):
                output = outputs[k]
                score = scores[k]
                att, emb = self.forward(source_embed, source_embed2,hv,ht, output, source_pad_mask=source_pad_mask,
                                        target_pad_mask=(output == pad_id))
                val, idx = torch.topk(att[:, -1, :], top_k)
                log_val = -torch.log(val)
                for i in range(top_k):
                    new_output = torch.cat([output, idx[:, i].view(-1, 1)], dim=-1)
                    new_score = score + log_val[:, i].view(-1)
                    possible_outputs.append(new_output.unsqueeze(0))
                    possible_scores.append(new_score.unsqueeze(0))
            possible_outputs = torch.cat(possible_outputs, dim=0)
            possible_scores = torch.cat(possible_scores, dim=0)
            val, idx = torch.topk(possible_scores, top_k, dim=0)
            col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)
            outputs = possible_outputs[idx, col_idx]
            scores = possible_scores[idx, col_idx]
        val, idx = torch.topk(scores, 1, dim=0)
        col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0], 1)
        output = outputs[idx, col_idx]
        return output.squeeze(0)

    def generate_square_subsequent_mask_with_source(self, src_sz, tgt_sz, mode='eye'):
        mask = self.generate_square_subsequent_mask(src_sz + tgt_sz)
        if mode == 'one':
            mask[:src_sz, :src_sz] = self.generate_square_mask(src_sz)
        elif mode == 'eye':
            mask[:src_sz, :src_sz] = self.generate_square_identity_mask(src_sz)
        else:
            raise ValueError('Mode must be "one" or "eye".')
        mask[src_sz:, src_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        return mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_identity_mask(self, sz):
        mask = (torch.eye(sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_mask(self, sz):
        mask = (torch.ones(sz, sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class Context(nn.Module):
    def __init__(self, classifier, generator, num_topics, embed_dim):
        super().__init__()
        self.classifier = classifier
        self.generator = generator
        self.label_embedding = nn.Embedding(num_topics, embed_dim)

    def forward(self, image, history=None, caption=None, label=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3,
                max_len=300, get_emb=False):
        label = label.long() if label != None else label
        img_mlc, img_emb, img_emb2,hv,ht = self.classifier(img=image, txt=history, lbl=label, threshold=threshold,
                                                                 pad_id=pad_id,get_embed=True)  # att, final_embed + emb, final_embed + emb2

        lbl_idx = torch.arange(img_emb.shape[1]).unsqueeze(0).repeat(img_emb.shape[0], 1).to(img_emb.device)
        lbl_emb = self.label_embedding(lbl_idx)
        lbl_idx2 = torch.arange(img_emb.shape[1]).unsqueeze(0).repeat(img_emb.shape[0], 1).to(img_emb.device)
        lbl_emb2 = self.label_embedding(lbl_idx2)

        #infer阶段没有caption
        if caption != None:
            src_emb = img_emb + lbl_emb
            src_emb2 = img_emb2 + lbl_emb2
            pad_mask = (caption == pad_id)
            cap_gen, cap_emb = self.generator(source_embed=src_emb, source_embed2=src_emb2,hv=hv,ht=ht, token_index=caption,
                                              target_pad_mask=pad_mask)
            if get_emb:
                # return cap_gen, img_mlc, cap_emb, loss_align
                return cap_gen, img_mlc, cap_emb
            else:
                # return cap_gen, img_mlc, loss_align
                return cap_gen, img_mlc
        else:
            src_emb = img_emb + lbl_emb
            src_emb2 = img_emb2 + lbl_emb2
            cap_gen = self.generator(source_embed=src_emb, source_embed2=src_emb2,hv=hv,ht=ht, token_index=caption, max_len=max_len,
                                     bos_id=bos_id, pad_id=pad_id)
            return cap_gen, img_mlc
