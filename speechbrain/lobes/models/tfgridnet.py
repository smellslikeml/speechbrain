"""This library provides the TF-GridNet model for speech separation.

TF-GridNet integrates full-band (intra-frame) and sub-band (inter-frame)
modeling in the time-frequency domain. It is a complete separator that
owns its STFT analysis and synthesis: it takes a mixture waveform and
directly returns one separated waveform per source, without predicting
masks over a learned encoder.

Reference: https://arxiv.org/abs/2211.12433
Reference: https://arxiv.org/abs/2209.03952

Ported from the official TF-GridNet implementation in ESPnet
(espnet2/enh/separator/tfgridnet_separator.py). Both ESPnet and
SpeechBrain are released under the Apache-2.0 license.

Authors
 * Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, S. Watanabe 2022
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def _get_activation(activation):
    """Returns an instance of a ``torch.nn`` activation module by name.

    Arguments
    ---------
    activation : str
        Case-insensitive name of a ``torch.nn`` activation class, e.g.
        ``"relu"`` or ``"prelu"``.

    Returns
    -------
    nn.Module
        The instantiated activation module.

    Example
    -------
    >>> act = _get_activation("prelu")
    >>> act(torch.rand(2, 4)).shape
    torch.Size([2, 4])
    """
    matches = [name for name in dir(nn) if name.lower() == activation.lower()]
    if len(matches) != 1 or not isinstance(getattr(nn, matches[0]), type):
        raise ValueError(f"Unknown activation function: {activation!r}")
    return getattr(nn, matches[0])()


class TFGridNet(nn.Module):
    """TF-GridNet separator, mapping a mixture waveform to its sources.

    The input mixture is normalized to unit variance, mapped to a (real,
    imaginary) spectrogram by an STFT, processed by a stack of
    GridNetBlocks, and mapped back to one waveform per source, where the
    normalization is undone. Like the reference implementation, the model
    works best when the separation loss is scale-invariant (e.g. SI-SNR).

    Arguments
    ---------
    n_srcs : int
        Number of sources/speakers to separate.
    n_fft : int
        STFT window size (in samples).
    stride : int
        STFT hop size (in samples).
    window : str
        STFT window type, e.g. "hann".
    n_imics : int
        Number of microphone channels (only fixed-array geometry
        is supported).
    n_layers : int
        Number of TF-GridNet blocks.
    lstm_hidden_units : int
        Number of hidden units in each LSTM.
    attn_n_head : int
        Number of heads in the self-attention.
    attn_approx_qk_dim : int
        Approximate dimension of the frame-level query/key tensors.
    emb_dim : int
        Embedding dimension.
    emb_ks : int
        Kernel size of the unfolding and the transposed 1D convolutions.
    emb_hs : int
        Hop size of the unfolding and the transposed 1D convolutions.
    activation : str
        Name of a ``torch.nn`` activation used throughout the model, e.g.
        "prelu".
    eps : float
        Small epsilon for the normalization layers.

    Example
    -------
    >>> model = TFGridNet(
    ...     n_srcs=2,
    ...     n_fft=64,
    ...     stride=32,
    ...     n_layers=1,
    ...     emb_dim=4,
    ...     lstm_hidden_units=8,
    ...     attn_n_head=1,
    ... )
    >>> mix = torch.rand(1, 1024)
    >>> model(mix).shape
    torch.Size([1, 2, 1024])
    """

    def __init__(
        self,
        n_srcs=2,
        n_fft=128,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        if not hasattr(torch, f"{window}_window"):
            raise ValueError(
                f"Unknown STFT window {window!r}: expected the name of a "
                "torch window function, e.g. 'hann' or 'hamming'"
            )

        self.n_srcs = n_srcs
        self.n_fft = n_fft
        self.stride = stride
        self.n_imics = n_imics
        self.n_layers = n_layers
        self.register_buffer(
            "stft_window", getattr(torch, f"{window}_window")(n_fft)
        )

        ks, padding = (3, 3), (1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList(
            [
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.deconv = nn.ConvTranspose2d(
            emb_dim, n_srcs * 2, ks, padding=padding
        )

    def forward(self, mix):
        """Separates a mixture waveform into its sources.

        Arguments
        ---------
        mix : torch.Tensor
            Mixture waveform with shape `(batch, time)` for single-channel
            audio, or `(batch, time, mics)` for multi-channel audio.

        Returns
        -------
        torch.Tensor
            Separated sources with shape `(batch, n_srcs, time)`, where the
            time dimension matches the input.

        Example
        -------
        >>> model = TFGridNet(n_layers=1, emb_dim=4, lstm_hidden_units=8)
        >>> mix = torch.rand(1, 1024)
        >>> model(mix).shape
        torch.Size([1, 2, 1024])
        """
        if mix.dim() == 2:
            wav = mix.unsqueeze(-1)
        elif mix.dim() == 3:
            wav = mix
        else:
            raise ValueError(
                f"Expected a 2- or 3-dimensional mixture, got {mix.dim()}"
            )
        if wav.shape[-1] != self.n_imics:
            raise ValueError(
                f"Expected {self.n_imics} channel(s), got {wav.shape[-1]}"
            )

        # the sources are separated at unit variance and rescaled at the end
        wav = wav.float()
        n_samples = wav.shape[1]
        mix_std = wav.std(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        wav = wav / mix_std

        # STFT analysis: [B*M, F, T] -> [B, M, T, F]
        spec = torch.stft(
            wav.permute(0, 2, 1).reshape(-1, n_samples),
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=self.stft_window.to(wav.dtype),
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        n_batch = wav.shape[0]
        n_frames = spec.shape[-1]
        n_freqs = spec.shape[-2]
        spec = spec.reshape(n_batch, self.n_imics, n_freqs, n_frames)
        spec = spec.transpose(2, 3)  # [B, M, T, F]
        feats = torch.cat((spec.real, spec.imag), dim=1)  # [B, 2*M, T, F]

        feats = self.conv(feats)  # [B, emb_dim, T, F]
        for block in self.blocks:
            feats = block(feats)
        feats = self.deconv(feats)  # [B, n_srcs*2, T, F]

        # STFT synthesis: [B, n_srcs, 2, T, F] -> [B, n_srcs, T]
        feats = feats.view(n_batch, self.n_srcs, 2, n_frames, n_freqs)
        spec = torch.complex(feats[:, :, 0], feats[:, :, 1])
        spec = spec.transpose(-1, -2)  # [B, n_srcs, F, T]
        est_src = torch.istft(
            spec.reshape(-1, n_freqs, n_frames),
            n_fft=self.n_fft,
            hop_length=self.stride,
            win_length=self.n_fft,
            window=self.stft_window.to(feats.dtype),
            center=True,
        ).view(n_batch, self.n_srcs, -1)

        # the ISTFT output can be a few samples shorter or longer than the
        # input: pad or trim it so that the output length is exact
        if est_src.shape[-1] < n_samples:
            est_src = F.pad(est_src, (0, n_samples - est_src.shape[-1]))
        else:
            est_src = est_src[..., :n_samples]

        est_src = est_src * mix_std  # undo the variance normalization
        return est_src.to(mix.dtype)


class GridNetBlock(nn.Module):
    """One TF-GridNet processing block.

    Each block runs, in order: an intra-frame (full-band) bidirectional
    LSTM over the frequency axis, an inter-frame (sub-band) bidirectional
    LSTM over the time axis, and a cross-frame multi-head self-attention,
    each with a residual connection.

    Arguments
    ---------
    emb_dim : int
        Embedding dimension (number of channels of the input map).
    emb_ks : int
        Kernel size of the unfolding and the transposed 1D convolution.
    emb_hs : int
        Hop size of the unfolding and the transposed 1D convolution.
    n_freqs : int
        Number of frequency bins of the input spectrogram.
    hidden_channels : int
        Number of hidden units in each LSTM.
    n_head : int
        Number of heads in the self-attention.
    approx_qk_dim : int
        Approximate dimension of the frame-level query/key tensors.
    activation : str
        Name of a ``torch.nn`` activation module, e.g. "prelu".
    eps : float
        Small epsilon for the normalization layers.

    Example
    -------
    >>> block = GridNetBlock(4, 4, 1, 33, 8)
    >>> x = torch.rand(1, 4, 20, 33)
    >>> block(x).shape
    torch.Size([1, 4, 20, 33])
    """

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels,
            hidden_channels,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels,
            hidden_channels,
            1,
            batch_first=True,
            bidirectional=True,
        )
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
        )

        # approx_qk_dim is only approximate
        qk_dim = math.ceil(approx_qk_dim / n_freqs)
        assert emb_dim % n_head == 0

        def attn_conv(out_dim):
            return nn.Sequential(
                nn.Conv2d(emb_dim, out_dim, 1),
                _get_activation(activation),
                LayerNormalization4DCF((out_dim, n_freqs), eps=eps),
            )

        self.attn_conv_Q = nn.ModuleList(
            [attn_conv(qk_dim) for _ in range(n_head)]
        )
        self.attn_conv_K = nn.ModuleList(
            [attn_conv(qk_dim) for _ in range(n_head)]
        )
        self.attn_conv_V = nn.ModuleList(
            [attn_conv(emb_dim // n_head) for _ in range(n_head)]
        )
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            _get_activation(activation),
            LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
        )
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        """Processes one TF-GridNet block.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map with shape `[B, C, T, Q]`, where `T` is the
            number of frames and `Q` the number of frequency bins.

        Returns
        -------
        torch.Tensor
            Output feature map with shape `[B, C, T, Q]`.
        """
        B, C, old_T, old_Q = x.shape
        # pad T and Q so that the (emb_ks, emb_hs) unfolding tiles them
        T = (
            math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        Q = (
            math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra-frame (full-band) RNN, run along the frequency axis
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        intra_rnn = F.unfold(
            intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # residual connection

        # inter-frame (sub-band) RNN, run along the time axis
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # [B, C, T, Q]
        inter_rnn = (
            inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
        )  # [BQ, C, T]
        inter_rnn = F.unfold(
            inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        )  # [BQ, C*emb_ks, -1]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, -1, C*emb_ks]
        inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BQ, -1, H]
        inter_rnn = inter_rnn.transpose(1, 2)  # [BQ, H, -1]
        inter_rnn = self.inter_linear(inter_rnn)  # [BQ, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
        inter_rnn = inter_rnn + input_  # residual connection

        # cross-frame multi-head self-attention over the time axis
        inter_rnn = inter_rnn[..., :old_T, :old_Q]
        feats = inter_rnn
        all_q = [conv(feats) for conv in self.attn_conv_Q]
        all_k = [conv(feats) for conv in self.attn_conv_K]
        all_v = [conv(feats) for conv in self.attn_conv_V]
        q = torch.cat(all_q, dim=0)  # [n_head*B, E, T, Q]
        k = torch.cat(all_k, dim=0)  # [n_head*B, E, T, Q]
        v = torch.cat(all_v, dim=0)  # [n_head*B, Ev, T, Q]

        q = q.transpose(1, 2).flatten(start_dim=2)  # [n_head*B, T, E*Q]
        k = k.transpose(1, 2).flatten(start_dim=2)  # [n_head*B, T, E*Q]
        v = v.transpose(1, 2)  # [n_head*B, T, Ev, Q]
        v_shape = v.shape
        v = v.flatten(start_dim=2)  # [n_head*B, T, Ev*Q]

        attn_dim = q.shape[-1]
        attn_mat = torch.matmul(q, k.transpose(1, 2)) / (attn_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)  # [n_head*B, T, T]
        v = torch.matmul(attn_mat, v)  # [n_head*B, T, Ev*Q]

        v = v.reshape(v_shape)  # [n_head*B, T, Ev, Q]
        v = v.transpose(1, 2)  # [n_head*B, Ev, T, Q]
        emb_dim = v.shape[1]
        feats = v.view([self.n_head, B, emb_dim, old_T, -1])
        feats = feats.transpose(0, 1).contiguous()  # [B, n_head, C, T, Q]
        feats = feats.view([B, self.n_head * emb_dim, old_T, -1])
        feats = self.attn_concat_proj(feats)  # [B, C, T, Q]

        return feats + inter_rnn


class LayerNormalization4D(nn.Module):
    """Layer normalization over the channel dimension of a `[B, C, T, Q]`
    tensor.

    Each time-frequency bin is normalized across channels, followed by a
    per-channel affine transform.

    Arguments
    ---------
    input_dimension : int
        Number of channels `C` of the expected input.
    eps : float
        Small epsilon added to the variance.

    Example
    -------
    >>> norm = LayerNormalization4D(4)
    >>> x = torch.rand(2, 4, 10, 12)
    >>> norm(x).shape
    torch.Size([2, 4, 10, 12])
    """

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.ones(*param_size))
        self.beta = Parameter(torch.zeros(*param_size))
        self.eps = eps

    def forward(self, x):
        """Normalizes the input across the channel dimension."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4 dimensions, but got {x.ndim}")
        mu = x.mean(dim=(1,), keepdim=True)  # [B, 1, T, Q]
        std = torch.sqrt(
            x.var(dim=(1,), unbiased=False, keepdim=True) + self.eps
        )  # [B, 1, T, Q]
        return (x - mu) / std * self.gamma + self.beta


class LayerNormalization4DCF(nn.Module):
    """Layer normalization over the channel and frequency dimensions of a
    `[B, C, T, Q]` tensor.

    Each time frame is normalized across channels and frequency bins,
    followed by a per-(channel, frequency) affine transform.

    Arguments
    ---------
    input_dimension : tuple
        The `(C, Q)` channel and frequency dimensions of the expected input.
    eps : float
        Small epsilon added to the variance.

    Example
    -------
    >>> norm = LayerNormalization4DCF((4, 12))
    >>> x = torch.rand(2, 4, 10, 12)
    >>> norm(x).shape
    torch.Size([2, 4, 10, 12])
    """

    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.ones(*param_size))
        self.beta = Parameter(torch.zeros(*param_size))
        self.eps = eps

    def forward(self, x):
        """Normalizes the input across channels and frequency bins."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4 dimensions, but got {x.ndim}")
        mu = x.mean(dim=(1, 3), keepdim=True)  # [B, 1, T, 1]
        std = torch.sqrt(
            x.var(dim=(1, 3), unbiased=False, keepdim=True) + self.eps
        )  # [B, 1, T, 1]
        return (x - mu) / std * self.gamma + self.beta
