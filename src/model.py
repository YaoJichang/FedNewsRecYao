import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel

import numpy as np

# 缩放点积注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k  # 每个头的维度

    def forward(self, Q, K, V, attn_mask=None):
        # 计算Q和K的点积，然后除以sqrt(d_k)进行缩放
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 应用softmax之前，对分数进行指数化
        scores = torch.exp(scores)
        # 如果有注意力掩码，则应用掩码
        if attn_mask is not None:
            scores = scores * attn_mask
        # 计算注意力权重
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)
        # 计算上下文向量
        context = torch.matmul(attn, V)
        return context, attn

# Top-K稀疏注意力机制
class TopK_SparseScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, top_k):
        super(TopK_SparseScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.top_k = top_k  # 每个查询向量保留的Top-K个分数

    def forward(self, Q, K, V, attn_mask=None):
        # 计算Q和K的点积，然后除以sqrt(d_k)进行缩放
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        # 对每个查询向量，保留Top-K个分数，其余置零
        top_k_scores = torch.zeros_like(scores)
        for i in range(scores.size(0)):  # 处理每个查询向量
            for j in range(scores.size(1)):  # 处理每个头
                # 获取当前查询向量的分数
                current_scores = scores[i, j, :, :]
                # 获取Top-K个分数的索引
                top_k_indices = torch.topk(current_scores, k=self.top_k, dim=-1).indices
                # 将Top-K个分数保留，其余置零
                top_k_scores[i, j, :, :].scatter_(-1, top_k_indices, current_scores.gather(-1, top_k_indices))

        # 应用softmax之前，对分数进行指数化
        sparse_scores = torch.exp(top_k_scores)

        # 如果有注意力掩码，则应用掩码
        if attn_mask is not None:
            sparse_scores = sparse_scores * attn_mask

        # 计算注意力权重
        attn = sparse_scores / (torch.sum(sparse_scores, dim=-1, keepdim=True) + 1e-8)

        # 计算上下文向量
        context = torch.matmul(attn, V)
        return context, attn

# 多头注意力机制
class TopK_SparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, top_k):
        super(TopK_SparseMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.top_k = top_k  # 每个查询向量保留的Top-K个分数

        # 定义Q, K, V的线性变换
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, attn_mask=None):
        batch_size, seq_len, _ = Q.size()

        # 将Q, K, V分别通过线性变换，并分割成多个头
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 如果有注意力掩码，则扩展掩码以匹配头的数量
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # 调用Top-K稀疏缩放点积注意力机制
        context, attn = TopK_SparseScaledDotProductAttention(self.d_k, self.top_k)(q_s, k_s, v_s, attn_mask)

        # 将多头的结果拼接起来
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        return context

# 加性注意力机制
class AdditiveAttention(nn.Module):
    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)  # 第一层全连接
        self.att_fc2 = nn.Linear(hidden_size, 1)    # 第二层全连接

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]  # 批量大小
        e = self.att_fc1(x)  # 第一层全连接
        e = nn.Tanh()(e)  # 应用Tanh激活函数
        alpha = self.att_fc2(e)  # 第二层全连接

        # 应用指数函数
        alpha = torch.exp(alpha)
        # 如果有注意力掩码，则应用掩码
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        # 归一化注意力权重
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        # 计算加权和
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # 重塑形状
        return x

# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self,
                 bert_type="bert-base-uncased",
                 word_embedding_dim=400,
                 dropout_rate=0.2,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_type,
                                              hidden_dropout_prob=0,
                                              attention_probs_dropout_prob=0)
        # 加性注意力机制
        self.additive_attention = AdditiveAttention(self.bert.config.hidden_size,
                                                    self.bert.config.hidden_size // 2)
        # 全连接层，将BERT的输出映射到目标维度
        self.fc = nn.Linear(self.bert.config.hidden_size, word_embedding_dim)

    def forward(self, text):
        # text的形状为(batch_size, 2, seq_len)，其中text[:,0,:]是tokens，text[:,1,:]是attention_mask
        tokens = text[:, 0, :]
        atts = text[:, 1, :]
        # 使用BERT模型提取文本的嵌入
        text_vector = self.bert(tokens, attention_mask=atts)[0]
        # 使用加性注意力机制提取文本的表示
        text_vector = self.additive_attention(text_vector)
        # 通过全连接层映射到目标维度
        text_vector = self.fc(text_vector)
        return text_vector

# 用户编码器
class UserEncoder(nn.Module):
    def __init__(self,
                 news_embedding_dim=400,
                 num_attention_heads=20,
                 query_vector_dim=200,
                 top_k=5):
        super(UserEncoder, self).__init__()
        self.dropout_rate = 0.2
        # 使用Top-K稀疏多头注意力机制
        self.multihead_attention = TopK_SparseMultiHeadAttention(news_embedding_dim,
                                                                num_attention_heads, 20, 20,
                                                                top_k)
        # 加性注意力机制
        self.additive_attention = AdditiveAttention(news_embedding_dim,
                                                    query_vector_dim)

    def forward(self, clicked_news_vecs):
        # 对点击历史进行dropout
        clicked_news_vecs = F.dropout(clicked_news_vecs, p=self.dropout_rate, training=self.training)
        # 使用Top-K稀疏多头注意力机制对点击历史进行编码
        multi_clicked_vectors = self.multihead_attention(
            clicked_news_vecs, clicked_news_vecs, clicked_news_vecs
        )
        # 使用加性注意力机制提取用户表示
        pos_user_vector = self.additive_attention(multi_clicked_vectors)
        user_vector = pos_user_vector
        return user_vector

# 推荐模型
class Model(nn.Module):
    def __init__(self, top_k=5, lambda_reg=0.001, mu=0.01):
        super(Model, self).__init__()
        # 用户编码器，使用Top-K稀疏多头注意力机制
        self.user_encoder = UserEncoder(top_k=top_k)
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.lambda_reg = lambda_reg  # 正则化系数
        self.mu = mu  # pFedMe的L2正则化系数

    def forward(self, candidate_vecs, clicked_news_vecs, targets, global_model=None, compute_loss=True):
        # 使用用户编码器生成用户表示
        user_vector = self.user_encoder(clicked_news_vecs)
        # 计算候选新闻和用户表示之间的匹配分数
        score = torch.bmm(candidate_vecs, user_vector.unsqueeze(-1)).squeeze(dim=-1)
        # 如果需要计算损失，则计算交叉熵损失
        if compute_loss:
            loss = self.criterion(score, targets)
            # pFedMe的L2正则化项
            if global_model is not None:
                l2_reg = 0.0
                for local_param, global_param in zip(self.user_encoder.parameters(), global_model.user_encoder.parameters()):
                    l2_reg += torch.sum((local_param - global_param) ** 2)
                loss += self.mu / 2 * l2_reg
            return loss, score
        else:
            return score