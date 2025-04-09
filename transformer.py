import math
import torch
import torch.nn as nn



class Embeddings(nn.Module):

    """ 

    Embeddings convert the the input tokens into the d_model dimension, and then it multiplies the embedded vector with the square root of the d_model
    in  Attention Is All You Need research paper the they used d_model = 512 i.e the dimension of the embeddings,
    
    """

    def __init__(self, d_model:int, vocab_size:int) -> None:
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionlEncoding(nn.Module):

    """
    
    The Positional Encodings helps model to use the order of the sequence so for that we should give the 
    model relative or absolute positions of the tokens in the sequence.

    it done by adding the Input Embeddings with the Positional Embeddings both having the same dimension as d_model

    for Positional Enocding there are many ways of doing it

      1. Learned Positional encoding
      2. Fixed Positional encoding 

    in  Attention Is All You Need they use the sine and cosine functions of different frequencies 
    
    for even index they used the sine functtion
        PE(pos,2i) = sin(pos/100002i/dmodel)

    and for the odd indexes they used the cosine function
        PE(pos,2i+1) = cos(pos/100002i/dmodel)
    
    because it allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float ) -> None:
        super(PositionlEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create the matrix of size (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create the vector of shape of seq len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * math.log(10000.0)/self.d_model)
        # Applying the sin function to the even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Applying the cosin function to the odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    """
    The Layer Normalization calculates the mean and standard deviation of the each item in an batch
    unlike the batch normalization were the we calculate the mean and standard deviation of the entire batch,
     and then we calculate the new values by using their own mean and standard deviation

      for that we introduce the two parameters that are the gamma(multiplicative) and beta(additive) 
      because maybe the it contains all the values in between 0-1 it may be restrictive to the network,
      by using these valuese the network will learn to tune thses two parameters (gamma & beta) to add fluctutations when necessary.
    """

    def __init__(self, eps: float = 10**-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std * self.eps) * self.beta


class FeedForwardBolck(nn.Module):

    """
    In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
    connected feed-forward network, which is applied to each position separately and identically. This
    consists of two linear transformations with a ReLU activation in between.
            FFN(x) = max(0,xW1 +b1)W2 +b2

    having,
     inner layer dimension(d_ff) = 2048,
     output layer dimension(d_model) = 512

    """

    def __init__(self, d_model: int, d_ff: int, dropout: float ) -> None:
        super(FeedForwardBolck, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    """
    Multi-head attention in transformers is a mechanism that splits the input data into multiple subspaces and
    applies independent attention processes to each subspace.
    Each "head" focuses on different aspects of the input, enabling the model to capture diverse relationships within the data simultaneously.
    These attention outputs are then concatenated and transformed through a linear layer to produce the final result.
    This approach enhances the model's ability to generalize by providing multiple perspectives of the input data.
    
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = dropout

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) #wk
        self.w_v = nn.Linear(d_model, d_model) #wv

        self.w_o = nn.Linear(d_model, d_model) # wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k =  query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):

    """
    Residual connections in transformer architectures address key challenges in training deep neural networks by
    preserving information and stabilizing gradient flow.
    These connections allow the original input to bypass each layer's transformations, ensuring that critical information isn't lost
    during processing. 
    By adding the layer's output to its input, residual connections mitigate the vanishing gradient problem,
    enabling effective backpropagation even in networks with dozens of layers.
    """

    def __init__(self,dropout: float) -> None:
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    """
    The encoder is composed of a stack of N = 6 identical layers. Each layer has two
    sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position
    wise fully connected feed-forward network. We employ a residual connection around each of
    the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is
    
    LayerNorm(x + Sublayer(x)),
    
    where Sublayer(x) is the function implemented by the sub-layer
    itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
    layers, produce outputs of dimension dmodel = 512
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBolck, dropout: float) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask) :
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList ) -> None:
        super(Encoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    """
    The decoder is also composed of a stack of N = 6 identical layers. In addition to the two
    sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head
    attention over the output of the encoder stack. Similar to the encoder, we employ residual connections
    around each of the sub-layers, followed by layer normalization. We also modify the self-attention
    sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This
    masking, combined with fact that the output embeddings are offset by one position, ensures that the
    predictions for position i can depend only on the known outputs at positions less than i.
    """

    def __init__(self,
            self_attention_block: MultiHeadAttentionBlock,
            cross_attention_block: MultiHeadAttentionBlock,
            feed_forward_bock: FeedForwardBolck,
            dropout: float
            ) -> None:

        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_bock
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x:self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x:self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
        

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self,x):
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):
    """
    Here we connects the different blocks of the transformer

    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: Embeddings, tgt_embed: Embeddings, src_pos: PositionlEncoding, tgt_pos: PositionlEncoding, projection_layer: ProjectionLayer) -> None:
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self,encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self,x):
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 20487) -> Transformer:
    
    # Embedding layers
    src_embed = Embeddings(d_model, src_vocab_size)
    tgt_embed = Embeddings(d_model, tgt_vocab_size)

    # Positional encoding
    src_pos = PositionlEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionlEncoding(d_model, tgt_seq_len, dropout)

    # encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBolck(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBolck(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection ayer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    #creating the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # initializing the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer