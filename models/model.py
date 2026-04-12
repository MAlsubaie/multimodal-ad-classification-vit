import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =====================================================================
# 3D Patch Embedding with Positional Encoding
# =====================================================================
class PatchEmbedding3D(nn.Module):
    """
    Convert 3D MRI volume into sequence of patch embeddings.
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, 1, H, W, D)
        x = self.proj(x)  # (B, embed_dim, H//ps, W//ps, D//ps)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class PositionalEncoding3D(nn.Module):
    """
    Learnable 3D positional encoding for spatial information.
    """
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


# =====================================================================
# Biomarker Encoder (with confidence predictor) + new DBA params
# =====================================================================
class BiomarkerEncoder(nn.Module):
    """
    Encode biomarkers and create learnable biomarker tokens + DBA coordinates.
    """
    def __init__(self, num_biomarkers=5, embed_dim=768, num_patches=512):
        super().__init__()
        self.num_biomarkers = num_biomarkers
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # Encode raw biomarkers to embeddings
        self.encoder = nn.Sequential(
            nn.Linear(num_biomarkers, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Learnable biomarker tokens (one per biomarker)
        self.bio_tokens = nn.Parameter(torch.randn(1, num_biomarkers, embed_dim) * 0.02)
        
        # Modulation network for token scaling
        self.modulator = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        # Spatial projector: biomarker embedding -> per-patch weights [0,1]
        self.spatial_projector = nn.Sequential(nn.Linear(embed_dim, num_patches), nn.Sigmoid())

        # Confidence predictor for uncertainty quantification
        self.confidence_predictor = nn.Sequential(
            nn.Linear(embed_dim, max(8, embed_dim // 4)),
            nn.GELU(),
            nn.Linear(max(8, embed_dim // 4), 1),
            nn.Sigmoid()
        )

        # -------------------------- DBA parameters --------------------------
        # Learnable base coordinates for each biomarker in normalized [-1, 1] space
        # shape: (num_biomarkers, 3) -> (z, y, x) ordering for grid_sample
        self.register_parameter('bio_base_coords', nn.Parameter(torch.randn(num_biomarkers, 3) * 0.1))

        # Offset predictor: from bio_embed -> per-biomarker 3 offsets (normalized)
        # We'll apply a tanh and small scale later
        self.offset_predictor = nn.Linear(embed_dim, num_biomarkers * 3)
        # optional learned scale for offsets (how far offsets may go)
        self.offset_scale = nn.Parameter(torch.tensor(0.2))  # small default displacement

    def forward(self, biomarkers):
        """
        Args:
            biomarkers: (B, num_biomarkers)
        Returns:
            bio_embed: (B, embed_dim)
            modulated_tokens: (B, num_biomarkers, embed_dim)
            spatial_map: (B, num_patches)
            confidence: (B, 1)
            base_coords: (num_biomarkers, 3)  # shared
            offsets: (B, num_biomarkers, 3)   # predicted per-sample offsets in normalized coords
        """
        bio_embed = self.encoder(biomarkers)  # (B, embed_dim)

        # modulated tokens
        bio_tokens = self.bio_tokens.expand(biomarkers.size(0), -1, -1)
        modulation = self.modulator(bio_embed).unsqueeze(1)
        modulated_tokens = bio_tokens * modulation

        # spatial map
        spatial_map = self.spatial_projector(bio_embed)  # (B, num_patches)

        # confidence
        confidence = self.confidence_predictor(bio_embed)  # (B, 1)

        # offsets (B, num_biomarkers*3) -> (B, num_biomarkers, 3)
        raw_offsets = self.offset_predictor(bio_embed)  # (B, num_biomarkers*3)
        offsets = torch.tanh(raw_offsets).view(-1, self.num_biomarkers, 3) * self.offset_scale

        base_coords = self.bio_base_coords  # (num_biomarkers, 3) shared across batch

        return bio_embed, modulated_tokens, spatial_map, confidence, base_coords, offsets


# =====================================================================
# Deformable Biomarker Attention (DBA)
# =====================================================================
class DeformableBiomarkerAttention(nn.Module):
    """
    Given a volumetric feature map (patch embeddings reshaped to D x H x W),
    sample per-biomarker feature vectors at learned base_coords + offsets using grid_sample,
    then run cross-attention where biomarker-derived queries attend to these sampled tokens.

    Returns: (B, N, embed_dim) contribution (expanded from biomarker-conditioned outputs)
    """
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # small projection in case sampled features need processing
        self.sample_proj = nn.Linear(embed_dim, embed_dim)

    def sample_features(self, patch_tokens, grid_size, base_coords, offsets):
        """
        patch_tokens: (B, N, C) excluding CLS (N = grid_size^3)
        grid_size: int (e.g., 8)
        base_coords: (num_biomarkers, 3) in [-1,1]
        offsets: (B, num_biomarkers, 3) in approx [-offset_scale, offset_scale]
        returns: sampled_tokens (B, num_biomarkers, C)
        """
        B, N, C = patch_tokens.shape
        assert N == grid_size ** 3, "N must equal grid_size^3"

        # reshape to (B, C, D, H, W) with D=H=W=grid_size
        feat_map = patch_tokens.transpose(1, 2).contiguous()  # (B, C, N)
        feat_map = feat_map.view(B, C, grid_size, grid_size, grid_size)  # (B, C, D, H, W)

        num_biomarkers = offsets.shape[1]

        # prepare base coords and offsets into grid_sample coordinates: grid_sample expects coords in [-1,1]
        # base_coords: (num_biomarkers, 3)
        # offsets: (B, num_biomarkers, 3)
        # desired sample coords = base_coords (broadcasted) + offsets -> (B, num_biomarkers, 3)

        # Expand base_coords to batch
        device = feat_map.device
        base = base_coords.to(device).unsqueeze(0).expand(B, -1, -1)  # (B, num_biomarkers, 3)
        sample_coords = base + offsets  # (B, num_biomarkers, 3)
        # clamp to [-1,1] to avoid sampling out of domain
        sample_coords = sample_coords.clamp(-1.0, 1.0)

        # grid_sample expects a grid of shape (B', D_out, H_out, W_out, 3)
        # We want one sample per biomarker -> make D_out=H_out=W_out=1 for each biomarker
        # We'll reshape to batch over (B * num_biomarkers)
        # Create grid per (B * num_biomarkers): shape (B*num_biomarkers, 1,1,1, 3)
        grid = sample_coords.view(B * num_biomarkers, 1, 1, 1, 3)

        # replicate feature map per biomarker into batch dimension to match grid
        # feat_map: (B, C, D, H, W) -> repeat per biomarker -> (B * num_biomarkers, C, D, H, W)
        feat_map_rep = feat_map.unsqueeze(1).expand(-1, num_biomarkers, -1, -1, -1, -1)
        feat_map_rep = feat_map_rep.contiguous().view(B * num_biomarkers, C, grid.shape[2] if False else feat_map.shape[2],
                                                     feat_map.shape[3], feat_map.shape[4])
        # Actually feat_map.shape[2..4] already correct; above line overly complex; let's just reshape properly:
        feat_map_rep = feat_map.unsqueeze(1).expand(-1, num_biomarkers, -1, -1, -1, -1)
        feat_map_rep = feat_map_rep.contiguous().view(B * num_biomarkers, C, feat_map.shape[2], feat_map.shape[3], feat_map.shape[4])

        # Use grid_sample to pick feature vectors
        # grid_sample input: (N, C, D, H, W), grid: (N, D_out, H_out, W_out, 3)
        # mode='bilinear' works for 5D with align_corners=True/False. Use align_corners=True for stability
        sampled = F.grid_sample(feat_map_rep, grid, mode='bilinear', padding_mode='border', align_corners=True)
        # sampled: (B * num_biomarkers, C, 1,1,1)
        sampled = sampled.view(B, num_biomarkers, C)  # (B, num_biomarkers, C)

        return sampled

    def forward(self, x, bio_embed, base_coords, offsets, grid_size, confidence=None):
        """
        x: (B, N+1, C) full token sequence including CLS
        bio_embed: (B, embed_dim)
        base_coords: (num_biomarkers, 3)
        offsets: (B, num_biomarkers, 3)
        grid_size: int (patch grid per side)
        confidence: (B, 1) optional
        returns: (B, N+1, C) contribution (expanded to every patch position)
        """
        # extract patch tokens (without cls)
        patch_tokens = x[:, 1:, :]  # (B, N, C) where N = grid_size^3

        # sample features per biomarker
        sampled_tokens = self.sample_features(patch_tokens, grid_size, base_coords, offsets)  # (B, num_biomarkers, C)

        # optionally project sampled tokens
        sampled_tokens = self.sample_proj(sampled_tokens)  # (B, num_biomarkers, C)

        # Prepare query: biomarker-derived query vector
        # bio_embed -> query shape (B, 1, C)
        bio_q = bio_embed.unsqueeze(1)  # (B, 1, C)

        # cross-attention: queries (bio_q) attend to sampled_tokens (K,V)
        # cross_attn expects (Q, K, V) where we pass batch_first=True so shapes are (B, L_q, C), (B, L_k, C)
        attn_out, _ = self.cross_attn(bio_q, sampled_tokens, sampled_tokens)  # (B, 1, C)
        # expand attn_out to all token positions (N+1)
        B, fullN, Cfull = x.shape
        attn_expanded = attn_out.expand(-1, fullN, -1)  # (B, N+1, C)

        # Optionally incorporate confidence by scaling
        if confidence is not None:
            attn_expanded = attn_expanded * confidence.unsqueeze(-1)  # (B, N+1, C)

        return attn_expanded


# =====================================================================
# Biomarker-Conditioned Attention with Uncertainty Fusion + DBA
# =====================================================================
class BiomarkerConditionedAttention(nn.Module):
    """
    Multi-head attention where biomarkers condition the attention mechanism,
    now enhanced with Deformable Biomarker Attention (DBA).
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1, num_biomarkers=5, num_patches=512):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Standard self-attention components
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # Biomarker conditioning components
        self.bio_query = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Deformable Biomarker Attention module (DBA)
        self.dba = DeformableBiomarkerAttention(embed_dim, num_heads)

        # Gating mechanism to blend (standard, conditioned, dba) -> use two-stage gating
        # We'll compute gating weights for conditioned vs standard, and for DBA vs conditioned
        self.gate = nn.Sequential(nn.Linear(embed_dim * 3, embed_dim), nn.Sigmoid())

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Store last attention maps for visualization (standard attention)
        self.last_attention_map = None

    def forward(self, x, bio_embed, confidence=None, base_coords=None, offsets=None):
        """
        Args:
            x: (B, N, embed_dim) - includes CLS token
            bio_embed: (B, embed_dim)
            confidence: (B, 1)
            base_coords: (num_biomarkers, 3)
            offsets: (B, num_biomarkers, 3)
        Returns:
            (B, N, embed_dim) - fused features
        """
        B, N, C = x.shape

        # ---------- Standard self-attention ----------
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, num_heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        standard_out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # store standard attention maps for visualization
        self.last_attention_map = attn.detach()

        # ---------- Biomarker-conditioned cross-attention (original path) ----------
        bio_q = self.bio_query(bio_embed).unsqueeze(1)  # (B, 1, C)
        conditioned_out, _ = self.cross_attn(bio_q, x, x)  # (B, 1, C)
        conditioned_out = conditioned_out.expand(-1, N, -1)  # (B, N, C)

        # ---------- Deformable Biomarker Attention path (DBA) ----------
        # DBA returns (B, N, C) contribution by sampling and cross-attending
        # Need grid_size: infer from x (exclude CLS)
        num_patches = x.shape[1] - 1
        grid_size = int(round(num_patches ** (1/3)))
        dba_out = self.dba(x, bio_embed, base_coords, offsets, grid_size, confidence)  # (B, N, C)

        # ---------- Gating: blend the three outputs ----------
        # gate_input: concat along channel dim
        gate_input = torch.cat([standard_out, conditioned_out, dba_out], dim=-1)  # (B, N, 3C)
        gate_weights = self.gate(gate_input)  # (B, N, C), values in (0,1)

        # We will interpret gate_weights as weight for (conditioned + dba) vs standard, but to incorporate DBA specifically,
        # combine as: out = gate_weights * (alpha * conditioned + (1-alpha) * dba) + (1-gate_weights)*standard_out
        # For simplicity, compute alpha per position as sigmoid on a small projection (reuse gate_weights as gating vector).
        # Let's create alpha by another linear on concatenated conditioned and dba (learnable implicitly by gate)
        alpha = 0.5  # fixed blend between conditioned_out and dba_out (could be learned, but keep simple)

        biomarker_path = alpha * conditioned_out + (1 - alpha) * dba_out  # (B, N, C)

        if confidence is not None:
            biomarker_path = biomarker_path * confidence.unsqueeze(-1)

        out = gate_weights * biomarker_path + (1 - gate_weights) * standard_out
        out = self.proj(out)
        return out


# =====================================================================
# Transformer Block
# =====================================================================
class TransformerBlock(nn.Module):
    """
    Transformer block with optional biomarker conditioning.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1, use_conditioning=False, num_biomarkers=5, num_patches=512):
        super().__init__()
        self.use_conditioning = use_conditioning
        self.norm1 = nn.LayerNorm(embed_dim)
        
        if use_conditioning:
            self.attn = BiomarkerConditionedAttention(embed_dim, num_heads, dropout, num_biomarkers, num_patches)
        else:
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, bio_embed=None, confidence=None, base_coords=None, offsets=None):
        # Attention with residual connection
        if self.use_conditioning and bio_embed is not None:
            x = x + self.attn(self.norm1(x), bio_embed, confidence, base_coords, offsets)
        else:
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
            x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


# =====================================================================
# Uncertainty-Aware Biomarker-Conditioned Vision Transformer
# =====================================================================
class BiomarkerConditionedViT(nn.Module):
    """
    Biomarker-Conditioned Vision Transformer for Multimodal Medical Classification
    with Deformable Biomarker Attention integrated.
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=1,
        num_classes=3,
        num_biomarkers=5,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        conditioning_layers=[4,5,6,7,8,9,10,11]
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding3D(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = PositionalEncoding3D(num_patches + 1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Biomarker encoder (now returns base_coords and offsets)
        self.bio_encoder = BiomarkerEncoder(num_biomarkers, embed_dim, num_patches)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, use_conditioning=(i in conditioning_layers),
                             num_biomarkers=num_biomarkers, num_patches=num_patches)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        self.conditioning_layers = conditioning_layers
        self.apply(self._init_weights)

        # Enable Bayesian dropout during inference for uncertainty (keep Dropout layers trainable)
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
                
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, mri, biomarkers):
        B = mri.shape[0]
        x = self.patch_embed(mri)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional encoding
        x = self.pos_embed(x)

        # Encode biomarkers -> get embedding, tokens, spatial map, confidence, base_coords, offsets
        bio_embed, bio_tokens, spatial_map, confidence, base_coords, offsets = self.bio_encoder(biomarkers)

        # Spatial Biomarker Modulation (apply to patch tokens only)
        x_patches = x[:, 1:, :] * spatial_map.unsqueeze(-1)  # (B, num_patches, C)
        x = torch.cat([x[:, :1, :], x_patches], dim=1)

        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            if i in self.conditioning_layers:
                # pass base_coords and offsets (DBA uses them)
                x = block(x, bio_embed, confidence, base_coords, offsets)
            else:
                x = block(x, None, None, None, None)

        x = self.norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits, confidence

    # Attention visualization
    def get_attention_maps(self, layer_indices=None):
        if layer_indices is None:
            layer_indices = self.conditioning_layers
        attention_maps = {}
        for i in layer_indices:
            if i < len(self.blocks) and hasattr(self.blocks[i].attn, 'last_attention_map'):
                if self.blocks[i].attn.last_attention_map is not None:
                    attention_maps[f'layer_{i}'] = self.blocks[i].attn.last_attention_map
        return attention_maps

    def get_spatial_biomarker_map(self, biomarkers):
        _, _, spatial_map, _, _, _ = self.bio_encoder(biomarkers)
        grid_size = int(round(spatial_map.shape[1] ** (1/3)))
        spatial_map_3d = spatial_map.view(-1, grid_size, grid_size, grid_size)
        return spatial_map_3d

    def predict_with_uncertainty(self, mri, biomarkers, n_samples=10):
        self.train()  # keep dropout active
        predictions = []
        confidences = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits, conf = self.forward(mri, biomarkers)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
                confidences.append(conf)
        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        mean_confidence = torch.stack(confidences).mean(dim=0)
        return mean_probs, std_probs, mean_confidence
        
    def compute_loss(self, logits, targets):
        """Compute the loss."""
        return self.criterion(logits, targets)


# =====================================================================
# Model Factory
# =====================================================================
def create_model(config=None, device=None, **kwargs):
    # Allow passing config or explicit kwargs. Fallback to defaults.
    num_classes = kwargs.get('num_classes', config['num_classes'] if config else 3)
    num_biomarkers = kwargs.get('num_biomarkers', 5)
    img_size = kwargs.get('image_size', config['image_size'] if config else 128)
    patch_size = kwargs.get('patch_size', 16)
    embed_dim = kwargs.get('hidden_size', config['hidden_size'] if config else 384)
    depth = kwargs.get('num_hidden_layers', config['num_hidden_layers'] if config else 12)
    num_heads = kwargs.get('num_attention_heads', config['num_attention_heads'] if config else 6)
    
    model = BiomarkerConditionedViT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,
        num_classes=num_classes,
        num_biomarkers=num_biomarkers,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        conditioning_layers=list(range(4, depth))
    )
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    print(f"✓ Model created on {device}")
    return model


# =====================================================================
# Testing Script
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Biomarker-Conditioned ViT with Deformable Biomarker Attention (DBA)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    print("\n📦 Creating model...")
    mm_model = create_model(device=device)

    total_params = sum(p.numel() for p in mm_model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Model Size: ~{total_params * 4 / 1024 / 1024:.2f} MB")

    print(f"\n🧪 Testing Forward Pass:")
    B = 2
    # smaller input for speed if running on CPU; original code uses 128^3, but that is heavy:
    # to test functionality quickly change img_size in create_biomarker_vit to 64 and patch_size accordingly if needed.
    # Here we respect your original shapes: 128^3 input (be careful on CPU).
    mri = torch.randn(B, 1, 128, 128, 128).to(device)
    biomarkers = torch.randn(B, 5).to(device)

    print(f"  Input MRI shape: {mri.shape}")
    print(f"  Input biomarkers shape: {biomarkers.shape}")

    with torch.no_grad():
        logits, confidence = mm_model(mri, biomarkers)
        probs = F.softmax(logits, dim=1)

    print(f"  Output logits shape: {logits.shape}")
    print(f"  Confidence scores: {confidence.squeeze().cpu().numpy()}")

    print(f"\n  Sample predictions:")
    for i in range(B):
        pred = probs[i].argmax().item()
        conf = probs[i].max().item()
        bio_conf = confidence[i].item()
        print(f"    Patient {i}: Class {pred} | Pred Confidence: {conf:.3f} | Biomarker Reliability: {bio_conf:.3f}")

    print(f"\n🎨 Testing Attention Visualization:")
    attention_maps = mm_model.get_attention_maps()
    print(f"  Extracted attention from {len(attention_maps)} layers")
    for layer_name, attn_map in attention_maps.items():
        print(f"    {layer_name}: shape {attn_map.shape} (B, heads, N, N)")

    print(f"\n🗺️  Testing Spatial Biomarker Map:")
    spatial_map_3d = mm_model.get_spatial_biomarker_map(biomarkers)
    print(f"  3D Spatial Map shape: {spatial_map_3d.shape} (B, grid, grid, grid)")

    print(f"\n📊 Testing Uncertainty Estimation (MC Dropout) with 3 samples for speed:")
    mean_probs, std_probs, mean_conf = mm_model.predict_with_uncertainty(mri, biomarkers, n_samples=3)
    print(f"  Mean probabilities shape: {mean_probs.shape}")
    print(f"  Uncertainty (std) shape: {std_probs.shape}")
    for i in range(B):
        pred = mean_probs[i].argmax().item()
        uncertainty = std_probs[i].max().item()
        print(f"    Patient {i}: Class {pred} | Uncertainty: {uncertainty:.3f}")

    print("\n✅ DBA-integrated tests completed.")
    print("=" * 70)
