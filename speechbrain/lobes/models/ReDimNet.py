"""This file implements the ReDimNet model from https://arxiv.org/abs/2407.18223

ReDimNet (Reshape Dimensions Network, Interspeech 2024) computes speaker
embeddings by interleaving 2D convolutional blocks operating on (channel,
frequency, time) feature maps and 1D attention blocks operating on
(channel * frequency, time) signals. Switching between the two views is a
lossless reshape that preserves the C * F * T volume, which lets every stage
aggregate all previously computed feature maps.

Pretrained weights for the reference implementation are released via
torch.hub (variants "b0"-"b3", "b5", "b6" and "m", embedding size 192, see
https://github.com/IDRnD/ReDimNet), but they are not loaded here: converting
those checkpoints to SpeechBrain naming is left as a separate task.

Authors
 * Ivan Yakovlev et al. 2024 (original model)
 * SpeechBrain port 2026
"""

# Adapted from the MIT-licensed reference implementation:
# https://github.com/IDRnD/ReDimNet (Copyright (c) 2024 ID R&D, Inc.)

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.ECAPA_TDNN import (
    AttentiveStatisticsPooling,
    BatchNorm1d,
    Conv1d,
)

# Size presets from the reference release: (mels, base channels,
# group divisor, stages). Each stage is (stride, num_blocks, conv_exp,
# att_block_red): "stride" pools the frequency axis, "num_blocks" is the
# number of 2D blocks, "conv_exp" expands channels inside the stage (squeezed
# back before returning to 1D), and "att_block_red" sets the hidden size of
# the 1D attention block to (channels * mels) // att_block_red (None: no 1D
# block). "m" is the reference default and matches "b2".
PRESETS = {
    "b0": (
        60,
        10,
        1,
        [
            (1, 2, 1, 30),
            (2, 3, 2, 30),
            (1, 3, 3, 30),
            (2, 4, 2, 10),
            (1, 3, 1, 10),
        ],
    ),
    "b1": (
        72,
        12,
        8,
        [
            (1, 2, 1, None),
            (2, 3, 1, None),
            (3, 4, 1, 12),
            (2, 5, 1, 12),
            (2, 3, 1, 8),
        ],
    ),
    "b2": (
        72,
        16,
        4,
        [
            (1, 2, 1, 12),
            (2, 2, 1, 12),
            (1, 3, 1, 12),
            (2, 4, 1, 8),
            (1, 4, 1, 8),
            (2, 4, 1, 4),
        ],
    ),
    "b3": (
        72,
        16,
        1,
        [
            (1, 6, 4, 32),
            (2, 6, 2, 32),
            (1, 8, 2, 32),
            (2, 10, 2, 16),
            (1, 10, 1, 16),
            (2, 8, 1, 16),
        ],
    ),
    "b4": (
        72,
        32,
        1,
        [
            (1, 4, 2, 48),
            (2, 4, 2, 48),
            (1, 6, 2, 48),
            (2, 6, 1, 32),
            (1, 8, 1, 24),
            (2, 4, 1, 16),
        ],
    ),
    "b5": (
        72,
        32,
        16,
        [
            (1, 4, 2, 48),
            (2, 4, 2, 48),
            (1, 6, 2, 48),
            (2, 6, 1, 32),
            (1, 8, 1, 24),
            (2, 4, 1, 16),
        ],
    ),
    "b6": (
        72,
        32,
        32,
        [
            (1, 4, 4, 32),
            (2, 6, 2, 32),
            (1, 6, 2, 24),
            (3, 8, 1, 24),
            (1, 8, 1, 16),
            (2, 8, 1, 16),
        ],
    ),
}
PRESETS["m"] = PRESETS["b2"]


class LayerNorm(nn.Module):
    """Layer normalization over the leading channel dimension of
    (batch, channels, ...) tensors ("channels_first" in the reference).

    Arguments
    ---------
    channels : int
        Number of channels to normalize.
    eps : float
        Epsilon value for numerical stability.

    Example
    -------
    >>> x = torch.rand([8, 16, 72, 120])
    >>> norm = LayerNorm(16)
    >>> norm(x).shape
    torch.Size([8, 16, 72, 120])
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.channels = (channels,)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        """Normalizes the input over its channel dimension."""
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        weight, bias = self.weight, self.bias
        for _ in range(x.ndim - 2):
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)
        return weight * x + bias


class to1d(nn.Module):
    """Reshapes a (batch, c, f, t) feature map into a (batch, c * f, t)
    signal (the inverse of `to2d`)."""

    def forward(self, x):
        """Returns the reshaped 1D signal."""
        bs, c, f, t = x.shape
        return x.permute(0, 2, 1, 3).reshape(bs, c * f, t)


class to2d(nn.Module):
    """Reshapes a (batch, c * f, t) signal into a (batch, c, f, t)
    feature map (the inverse of `to1d`).

    Arguments
    ---------
    f : int
        Frequency size of the output feature map.
    c : int
        Channel size of the output feature map.

    Example
    -------
    >>> x = torch.rand([8, 16 * 72, 120])
    >>> reshape = to2d(72, 16)
    >>> reshape(x).shape
    torch.Size([8, 16, 72, 120])
    """

    def __init__(self, f, c):
        super().__init__()
        self.f = f
        self.c = c

    def forward(self, x):
        """Returns the reshaped 2D feature map."""
        bs, _, t = x.shape
        return x.reshape(bs, self.f, self.c, t).permute(0, 2, 1, 3)


class Weight1d(nn.Module):
    """Learnable weighted aggregation of equally-shaped 1D feature maps
    (`weigth1d` in the reference implementation). The aggregation weights
    are softmax-normalized over the feature maps, separately per channel.

    Arguments
    ---------
    n_maps : int
        Number of feature maps to aggregate.
    channels : int
        Number of channels of every feature map.

    Example
    -------
    >>> x = torch.rand([8, 64, 120])
    >>> agg = Weight1d(3, 64)
    >>> agg([x, x, x]).shape
    torch.Size([8, 64, 120])
    """

    def __init__(self, n_maps, channels):
        super().__init__()
        self.n_maps = n_maps
        if n_maps == 1:
            # Aggregating a single map is the identity.
            self.w = nn.Parameter(
                torch.ones(1, 1, channels, 1), requires_grad=False
            )
        else:
            self.w = nn.Parameter(torch.zeros(1, n_maps, channels, 1))

    def forward(self, xs):
        """Aggregates a list of feature maps into a single feature map."""
        if self.n_maps == 1:
            return xs[0]
        x = torch.stack(xs, dim=1)
        return (F.softmax(self.w, dim=1) * x).sum(dim=1)


class ConvNeXtLikeBlock(nn.Module):
    """A ConvNeXt-like residual block: depthwise convolution, batch
    normalization, activation, and a pointwise convolution.

    Arguments
    ---------
    channels : int
        Number of input and output channels.
    kernel_size : tuple
        Kernel size of the depthwise convolution.
    group_divisor : int
        The depthwise convolution uses channels // group_divisor groups.

    Example
    -------
    >>> x = torch.rand([8, 16, 72, 120])
    >>> block = ConvNeXtLikeBlock(16, group_divisor=4)
    >>> block(x).shape
    torch.Size([8, 16, 72, 120])
    """

    def __init__(self, channels, kernel_size=(3, 3), group_divisor=1):
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding="same",
            groups=max(channels // group_divisor, 1),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        return x + self.pwconv(self.act(self.norm(self.dwconv(x))))


class PosEncConv(nn.Module):
    """Convolutional positional encoding: a depthwise convolution followed
    by layer normalization, applied as a residual.

    Arguments
    ---------
    channels : int
        Number of input and output channels.
    kernel_size : int
        Kernel size of the depthwise convolution (odd).

    Example
    -------
    >>> x = torch.rand([8, 64, 120])
    >>> pos_enc = PosEncConv(64, kernel_size=59)
    >>> pos_enc(x).shape
    torch.Size([8, 64, 120])
    """

    def __init__(self, channels, kernel_size=59):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        return x + self.norm(self.conv(x))


class TransformerEncoderLayer(nn.Module):
    """A post-normalization transformer encoder layer with a fixed number
    of attention heads, operating on (batch, channels, time) tensors.

    Arguments
    ---------
    n_state : int
        Input/output size (must be divisible by n_head).
    n_mlp : int
        Hidden size of the feed-forward module.
    n_head : int
        Number of attention heads.

    Example
    -------
    >>> x = torch.rand([8, 64, 120])
    >>> layer = TransformerEncoderLayer(64, 128)
    >>> layer(x).shape
    torch.Size([8, 64, 120])
    """

    def __init__(self, n_state, n_mlp, n_head=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            n_state, n_head, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(n_state, eps=1e-6)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_state, n_mlp),
            nn.GELU(),
            nn.Linear(n_mlp, n_state),
        )
        self.final_layer_norm = nn.LayerNorm(n_state, eps=1e-6)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        x = x.transpose(1, 2)
        x = x + self.attention(x, x, x, need_weights=False)[0]
        x = self.layer_norm(x)
        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)
        return x.transpose(1, 2)


class TimeContextBlock1d(nn.Module):
    """A bottleneck-residual block that models time context with a
    convolutional positional encoding and self-attention (the "att" block
    of the reference implementation).

    Arguments
    ---------
    channels : int
        Number of input and output channels.
    hidden_channels : int
        Bottleneck size (must be divisible by 4, the number of heads).
    pos_ker_sz : int
        Kernel size of the positional-encoding convolution.

    Example
    -------
    >>> x = torch.rand([8, 1152, 120])
    >>> block = TimeContextBlock1d(1152, 96)
    >>> block(x).shape
    torch.Size([8, 1152, 120])
    """

    def __init__(self, channels, hidden_channels, pos_ker_sz=59):
        super().__init__()
        assert hidden_channels % 4 == 0, (
            "hidden_channels must be divisible by the 4 attention heads"
        )
        self.red_dim_conv = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1),
            LayerNorm(hidden_channels, eps=1e-6),
        )
        self.tcm = nn.Sequential(
            PosEncConv(hidden_channels, pos_ker_sz),
            TransformerEncoderLayer(hidden_channels, 2 * hidden_channels),
        )
        self.exp_dim_conv = nn.Conv1d(hidden_channels, channels, 1)

    def forward(self, x):
        """Processes the input tensor x and returns an output tensor."""
        return x + self.exp_dim_conv(self.tcm(self.red_dim_conv(x)))


class ReDimNet(torch.nn.Module):
    """An implementation of the speaker embedding model in the paper
    "ReDimNet: Reshape Dimensions Network for Speaker Recognition"
    (https://arxiv.org/abs/2407.18223), ported from the MIT-licensed
    reference implementation (https://github.com/IDRnD/ReDimNet).

    Unlike the reference model, features are expected as an input (mel
    filterbanks are computed in the recipe, as for ECAPA-TDNN), the
    pooling head reuses the lengths-aware ``AttentiveStatisticsPooling``
    from :py:class:`~speechbrain.lobes.models.ECAPA_TDNN`, and every
    preset pairs its stages with ConvNeXt-like 2D blocks (the reference
    also ships "basic_resnet" 2D blocks and "conv+att" 1D blocks for
    some presets, so parameter counts differ from the released models).

    Arguments
    ---------
    input_size : int
        Expected size of the input dimension (number of mel filters).
        It must be divisible by the product of the preset's stage
        strides (all presets use 72 mels except "b0", which uses 60).
    device : str
        Device used, e.g., "cpu" or "cuda".
    lin_neurons : int
        Number of neurons in the linear layer (embedding size).
    model_name : str
        Size preset, one of "b0"-"b6" or "m" (the reference default,
        identical to "b2").
    attention_channels : int
        The number of attention channels in the pooling layer.
    global_context : bool
        Whether to use global context in the pooling layer.
    out_channels : int
        If given, the aggregated feature maps are projected to this many
        channels before pooling (all reference presets use None, which
        keeps the C * F channels).

    Example
    -------
    >>> input_feats = torch.rand([5, 200, 72])
    >>> compute_embedding = ReDimNet(72, lin_neurons=192, model_name="m")
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    >>> torch.isfinite(outputs).all()
    tensor(True)
    """

    def __init__(
        self,
        input_size,
        device="cpu",
        lin_neurons=192,
        model_name="m",
        attention_channels=128,
        global_context=True,
        out_channels=None,
    ):
        super().__init__()
        if model_name.lower() not in PRESETS:
            raise ValueError(
                f"Unknown model_name {model_name!r}, expected one of "
                f"{sorted(PRESETS)}"
            )
        _, channels, group_divisor, stages_setup = PRESETS[model_name.lower()]
        c, f = channels, input_size

        # Stem: convolve the mel map, then flatten it to a 1D signal
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding="same"),
            LayerNorm(c, eps=1e-6),
            to1d(),
        )

        # Stages: aggregate all prior 1D feature maps, reshape to 2D,
        # pool frequency while expanding channels, apply the 2D blocks,
        # then reshape back and apply a 1D time-context block. Every
        # stage preserves the C * F volume.
        self.feature_weights = nn.ModuleList()
        self.stages = nn.ModuleList()
        for stage_id, (
            stride,
            num_blocks,
            conv_exp,
            att_block_red,
        ) in enumerate(stages_setup):
            assert f % stride == 0, (
                "input_size must be divisible by the product of the "
                "preset's stage strides"
            )
            self.feature_weights.append(Weight1d(stage_id + 1, c * f))
            stage = [to2d(f, c)]
            stage.append(
                nn.Conv2d(
                    c,
                    int(stride * c * conv_exp),
                    kernel_size=(stride, 1),
                    stride=(stride, 1),
                )
            )
            c = int(stride * c)
            f = f // stride
            stage.extend(
                ConvNeXtLikeBlock(
                    int(c * conv_exp), group_divisor=group_divisor
                )
                for _ in range(num_blocks)
            )
            if conv_exp != 1:
                # Squeeze the channel expansion back to c channels
                stage.append(
                    nn.Sequential(
                        nn.Conv2d(
                            int(c * conv_exp),
                            c,
                            kernel_size=(3, 3),
                            padding="same",
                            groups=max(c // group_divisor, 1),
                        ),
                        nn.BatchNorm2d(c, eps=1e-6),
                        nn.GELU(),
                        nn.Conv2d(c, c, kernel_size=1),
                    )
                )
            stage.append(to1d())
            if att_block_red is not None:
                stage.append(
                    TimeContextBlock1d(c * f, (c * f) // att_block_red)
                )
            self.stages.append(nn.Sequential(*stage))

        # Final aggregation of all stage outputs
        self.fin_wght = Weight1d(len(stages_setup) + 1, c * f)
        if out_channels is not None:
            self.mfa = nn.Sequential(
                Conv1d(
                    in_channels=c * f, out_channels=out_channels, kernel_size=1
                ),
                BatchNorm1d(input_size=out_channels),
            )
        else:
            self.mfa = nn.Identity()

        # Attentive statistics pooling + embedding (as in ECAPA-TDNN)
        head_in = out_channels if out_channels is not None else c * f
        self.asp = AttentiveStatisticsPooling(
            head_in,
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=head_in * 2)
        self.fc = Conv1d(
            in_channels=head_in * 2, out_channels=lin_neurons, kernel_size=1
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        lengths : torch.Tensor
            Corresponding relative lengths of the inputs.

        Returns
        -------
        x : torch.Tensor
            Embedding vector of shape (batch, 1, lin_neurons).
        """
        # (batch, time, feats) -> (batch, 1, feats, time) mel map
        outputs_1d = [self.stem(x.transpose(1, 2).unsqueeze(1))]
        for weight, stage in zip(self.feature_weights, self.stages):
            outputs_1d.append(stage(weight(outputs_1d)))
        x = self.fin_wght(outputs_1d)
        x = self.mfa(x)

        # Attentive statistics pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        return x.transpose(1, 2)
