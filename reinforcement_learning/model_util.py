

import torch.nn as nn
import torch


class CCT(nn.Module):
    """
        Compact Convolutional Transformer (CCT) Model
        https://arxiv.org/abs/2104.05704v4
    """

    def __init__(
        self,
        conv_kernel: int = 3, conv_stride: int = 2, conv_pad: int = 3,
        pool_kernel: int = 3, pool_stride: int = 2, pool_pad: int = 1,
        heads: int = 4, emb_dim: int = 256, feat_dim: int = 2*256,
        dropout: float = 0.0, attention_dropout: float = 0.0, layers: int = 7,
        channels: int = 3, image_size: int = 32, num_class: int = 256
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.image_size = image_size

        self.tokenizer = ConvTokenizer(
            channels=channels, emb_dim=self.emb_dim,
            conv_kernel=conv_kernel, conv_stride=conv_stride, conv_pad=conv_pad,
            pool_kernel=pool_kernel, pool_stride=pool_stride, pool_pad=pool_pad,
            activation=nn.ReLU
        )

        with torch.no_grad():
            x = torch.randn([1, channels, image_size, image_size])
            out = self.tokenizer(x)
            _, _, ph_c, pw_c = out.shape

        self.linear_projection = nn.Linear(
            ph_c, pw_c, self.emb_dim
        )

        self.pos_emb = nn.Parameter(
            torch.randn(
                [1, ph_c*pw_c, self.emb_dim]
            ).normal_(std=0.02)  # from torchvision, which takes this from BERT
        )
        self.dropout = nn.Dropout(dropout)
        encoders = []
        for _ in range(0, layers):
            encoders.append(
                TransformerEncoderBlock(
                    n_h=heads, emb_dim=self.emb_dim, feat_dim=feat_dim,
                    dropout=dropout, attention_dropout=attention_dropout
                )
            )
        self.encoder_stack = nn.Sequential(*encoders)
        self.seq_pool = SeqPool(emb_dim=self.emb_dim)
        self.mlp_head = nn.Linear(self.emb_dim, num_class)

    def forward(self, x: torch.Tensor):
        bs, c, h, w = x.shape  # (bs, c, h, w)

        # Creates overlapping patches using ConvNet
        x = self.tokenizer(x)
        x = rearrange(
            x, 'bs e_d ph_h ph_w -> bs (ph_h ph_w) e_d',
            bs=bs, e_d=self.emb_dim
        )

        # Add position embedding
        x = self.pos_emb.expand(bs, -1, -1) + x
        x = self.dropout(x)

        # Pass through Transformer Encoder layers
        x = self.encoder_stack(x)

        # Perform Sequential Pooling <- Novelty of the paper
        x = self.seq_pool(x)

        # MLP head used to get logits
        x = self.mlp_head(x)

        return x
