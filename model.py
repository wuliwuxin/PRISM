"""
PRISM: Primitive-based Recurrent Inference for Sequence Modeling
Model Architecture Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Patch-wise embedding for time series"""

    def __init__(self, patch_len=16, stride=8, d_model=512, in_channels=1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

        self.value_embedding = nn.Linear(patch_len * in_channels, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, 500, d_model))

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, channels]
        Returns:
            patches: [batch, n_patches, d_model]
            n_patches: int
        """
        batch_size, seq_len, channels = x.shape

        # Create patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i + self.patch_len, :]
            patches.append(patch)

        if len(patches) == 0:
            patches.append(x[:, :self.patch_len, :])

        patches = torch.stack(patches, dim=1)
        n_patches = patches.shape[1]
        patches = patches.reshape(batch_size, n_patches, -1)
        patches = self.value_embedding(patches)

        # Add positional encoding
        if n_patches <= self.position_embedding.shape[1]:
            patches = patches + self.position_embedding[:, :n_patches, :]
        else:
            pos_emb = self.position_embedding.repeat(1, (n_patches // 500) + 1, 1)[:, :n_patches, :]
            patches = patches + pos_emb

        return patches, n_patches


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention mechanism"""

    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(output)

        return output


class PrimitiveDictionaryDecomposition(nn.Module):
    """
    Primitive Dictionary Decomposition Module
    Decomposes time series into learnable primitive patterns
    """

    def __init__(self, d_model=512, n_primitives=16, n_heads=8, dropout=0.1):
        super(PrimitiveDictionaryDecomposition, self).__init__()
        self.d_model = d_model
        self.n_primitives = n_primitives
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Learnable primitive queries
        self.primitive_queries = nn.Parameter(torch.randn(n_primitives, d_model))

        # Projection layers
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def compute_diversity_loss(self, primitive_features):
        """Encourage diversity among primitives"""
        batch_size = primitive_features.shape[0]
        prim_norm = F.normalize(primitive_features, p=2, dim=-1)
        similarity = torch.bmm(prim_norm, prim_norm.transpose(1, 2))
        mask = torch.eye(self.n_primitives, device=similarity.device).unsqueeze(0)
        similarity = similarity * (1 - mask)
        diversity_loss = similarity.abs().mean()
        return diversity_loss

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
            diversity_loss: scalar
            primitive_weights: [batch, n_primitives]
        """
        batch_size, seq_len, _ = x.shape

        queries = self.primitive_queries.unsqueeze(0).expand(batch_size, -1, -1)
        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Multi-head attention
        Q = queries.view(batch_size, self.n_primitives, self.n_heads, self.d_k).transpose(1, 2)
        K = keys.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = values.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Primitive features
        primitive_features = torch.matmul(attn_weights, V)
        primitive_features = primitive_features.transpose(1, 2).contiguous().view(
            batch_size, self.n_primitives, self.d_model
        )

        # Global weights
        global_scores = torch.mean(scores, dim=(1, 3))
        primitive_weights = F.softmax(global_scores, dim=-1)

        # Aggregated representation
        aggregated = torch.bmm(primitive_weights.unsqueeze(1), primitive_features).squeeze(1)
        output = aggregated.unsqueeze(1).expand(-1, seq_len, -1)
        output = self.layer_norm(x + self.dropout(self.out_proj(output)))

        diversity_loss = self.compute_diversity_loss(primitive_features)

        return output, diversity_loss, primitive_weights


class AdaptiveSpectralRefinement(nn.Module):
    """
    Adaptive Spectral Refinement Module
    Refines representations in frequency domain
    """

    def __init__(self, seq_len=96, d_model=512, dropout=0.1):
        super(AdaptiveSpectralRefinement, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Learnable frequency weights
        freq_dim = seq_len // 2 + 1
        self.freq_weights = nn.Parameter(torch.ones(freq_dim))

        # Low and high frequency projections
        self.low_freq_proj = nn.Linear(d_model, d_model)
        self.high_freq_proj = nn.Linear(d_model, d_model)

        # Adaptive gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        residual = x
        batch_size = x.shape[0]

        # FFT for each feature dimension
        x_freq_list = []
        for i in range(self.d_model):
            x_i = x[:, :, i]
            x_freq_i = torch.fft.rfft(x_i, dim=1)
            x_freq_list.append(x_freq_i)

        x_freq = torch.stack(x_freq_list, dim=-1)

        # Apply learnable frequency weights
        freq_weights_expanded = self.freq_weights.unsqueeze(0).unsqueeze(-1)
        x_freq_weighted = x_freq * freq_weights_expanded

        # Split into low and high frequency
        cutoff = x_freq.shape[1] // 3
        x_freq_low = x_freq_weighted.clone()
        x_freq_high = x_freq_weighted.clone()

        x_freq_low[:, cutoff:, :] = 0
        x_freq_high[:, :cutoff, :] = 0

        # Inverse FFT to time domain
        x_low_list = []
        x_high_list = []
        for i in range(self.d_model):
            x_low_i = torch.fft.irfft(x_freq_low[:, :, i], n=self.seq_len, dim=1)
            x_high_i = torch.fft.irfft(x_freq_high[:, :, i], n=self.seq_len, dim=1)
            x_low_list.append(x_low_i)
            x_high_list.append(x_high_i)

        x_low = torch.stack(x_low_list, dim=-1)
        x_high = torch.stack(x_high_list, dim=-1)

        # Apply projections
        x_low = self.low_freq_proj(x_low)
        x_high = self.high_freq_proj(x_high)

        # Adaptive gating
        gate = self.fusion_gate(torch.cat([x_low, x_high], dim=-1))
        output = gate * x_low + (1 - gate) * x_high

        # Residual connection and normalization
        output = self.layer_norm(residual + self.dropout(output))

        return output


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class PRISMEncoderLayer(nn.Module):
    """PRISM Encoder Layer combining all components"""

    def __init__(self, seq_len=96, d_model=512, n_heads=8, d_ff=2048,
                 n_primitives=16, dropout=0.1):
        super(PRISMEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.primitive_decomp = PrimitiveDictionaryDecomposition(
            d_model, n_primitives, n_heads, dropout
        )

        self.spectral_refinement = AdaptiveSpectralRefinement(seq_len, d_model, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Primitive Dictionary Decomposition
        x, diversity_loss, primitive_weights = self.primitive_decomp(x)

        # Adaptive Spectral Refinement
        x = self.spectral_refinement(x)

        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x, diversity_loss, primitive_weights


class PRISM(nn.Module):
    """
    PRISM: Primitive-based Recurrent Inference for Sequence Modeling
    Complete model architecture
    """

    def __init__(
            self,
            seq_len=96,
            pred_len=24,
            use_patch=True,
            patch_len=16,
            stride=8,
            d_model=512,
            n_heads=8,
            e_layers=3,
            d_ff=2048,
            n_primitives=16,
            dropout=0.1,
    ):
        super(PRISM, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.use_patch = use_patch

        # Input embedding
        if use_patch:
            self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, in_channels=1)
            self.effective_seq_len = (seq_len - patch_len) // stride + 1
        else:
            self.enc_embedding = nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            self.effective_seq_len = seq_len

        # Time embeddings
        self.hour_embedding = nn.Embedding(24, d_model // 4)
        self.day_embedding = nn.Embedding(7, d_model // 4)
        self.month_embedding = nn.Embedding(12, d_model // 4)
        self.weekend_embedding = nn.Embedding(2, d_model // 4)
        self.time_proj = nn.Linear(d_model, d_model)

        # PRISM encoder layers
        self.encoder_layers = nn.ModuleList([
            PRISMEncoderLayer(self.effective_seq_len, d_model, n_heads, d_ff, n_primitives, dropout)
            for _ in range(e_layers)
        ])

        # Prediction head
        if use_patch:
            self.projection = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(d_model * self.effective_seq_len, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, pred_len)
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, pred_len)
            )

        self.use_norm = True

    def forward(self, x, hour_of_day, day_of_week, month=None, is_weekend=None,
                return_primitives=False):
        """
        Forward pass

        Args:
            x: [batch, seq_len, 1]
            hour_of_day: [batch, seq_len]
            day_of_week: [batch, seq_len]
            month: [batch, seq_len]
            is_weekend: [batch, seq_len]
            return_primitives: whether to return primitive weights

        Returns:
            predictions: [batch, pred_len]
            diversity_loss_total: scalar
            primitive_weights_list: list of [batch, n_primitives] (optional)
        """
        batch_size = x.shape[0]

        # Normalization
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
        else:
            x_norm = x
            means = torch.zeros_like(x[:, 0:1, :])
            stdev = torch.ones_like(x[:, 0:1, :])

        # Embedding
        if self.use_patch:
            x_emb, n_patches = self.patch_embedding(x_norm)

            patch_indices = list(range(0, self.seq_len - self.patch_embedding.patch_len + 1,
                                       self.patch_embedding.stride))
            patch_indices = patch_indices[:n_patches]

            if len(patch_indices) < n_patches:
                patch_indices.extend([self.seq_len - 1] * (n_patches - len(patch_indices)))

            patch_indices = torch.tensor(patch_indices, device=x.device)
            hour_patches = hour_of_day[:, patch_indices]
            day_patches = day_of_week[:, patch_indices]

            hour_emb = self.hour_embedding(hour_patches)
            day_emb = self.day_embedding(day_patches)

            if month is not None and is_weekend is not None:
                month_patches = month[:, patch_indices]
                weekend_patches = is_weekend[:, patch_indices]
                month_emb = self.month_embedding(month_patches)
                weekend_emb = self.weekend_embedding(weekend_patches)
            else:
                month_emb = torch.zeros_like(hour_emb)
                weekend_emb = torch.zeros_like(hour_emb)

            time_feat = torch.cat([hour_emb, day_emb, month_emb, weekend_emb], dim=-1)
        else:
            x_emb = self.enc_embedding(x_norm)

            hour_emb = self.hour_embedding(hour_of_day)
            day_emb = self.day_embedding(day_of_week)

            if month is not None and is_weekend is not None:
                month_emb = self.month_embedding(month)
                weekend_emb = self.weekend_embedding(is_weekend)
            else:
                month_emb = torch.zeros_like(hour_emb)
                weekend_emb = torch.zeros_like(hour_emb)

            time_feat = torch.cat([hour_emb, day_emb, month_emb, weekend_emb], dim=-1)

        # Fuse time features
        time_feat = self.time_proj(time_feat)
        x_emb = x_emb + time_feat

        # Encoder layers
        diversity_loss_total = 0.0
        primitive_weights_list = []

        for layer in self.encoder_layers:
            x_emb, diversity_loss, primitive_weights = layer(x_emb)
            diversity_loss_total += diversity_loss
            if return_primitives:
                primitive_weights_list.append(primitive_weights)

        # Prediction
        if self.use_patch:
            predictions = self.projection(x_emb)
        else:
            x_pooled = torch.mean(x_emb, dim=1)
            predictions = self.projection(x_pooled)

        # Denormalization
        if self.use_norm:
            predictions = predictions * stdev.squeeze(1) + means.squeeze(1)

        if return_primitives:
            return predictions, diversity_loss_total, primitive_weights_list

        return predictions, diversity_loss_total


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable