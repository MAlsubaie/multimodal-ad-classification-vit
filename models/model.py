import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def validate_config(config):
    """Validate configuration parameters."""
    required_keys = [
        "image_size", "num_channels", "num_classes", "hidden_size", 
        "num_hidden_layers", "num_attention_heads", "intermediate_size"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate dimensions
    if config["hidden_size"] % config["num_attention_heads"] != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    
    if config["image_size"] <= 0 or not isinstance(config["image_size"], int):
        raise ValueError("image_size must be a positive integer")
    
    if config["num_classes"] <= 0:
        raise ValueError("num_classes must be positive")

def validate_input_tensor(x, expected_channels, expected_size):
    """Validate input tensor dimensions."""
    if x.dim() != 5:
        raise ValueError(f"Expected 5D tensor (B, C, D, H, W), got {x.dim()}D")
    
    if x.size(1) != expected_channels:
        raise ValueError(f"Expected {expected_channels} channels, got {x.size(1)}")
    
    if x.size(2) != expected_size or x.size(3) != expected_size or x.size(4) != expected_size:
        raise ValueError(f"Expected size {expected_size}x{expected_size}x{expected_size}, got {x.size(2)}x{x.size(3)}x{x.size(4)}")

# --- Improved Activations ---
class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    More stable than standard GELU with better numerical precision.
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

# --- DropPath (Stochastic Depth) ---
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# --- Improved ConvBlock ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool3d(kernel_size=pool_size)
        )

    def forward(self, x):
        return self.block(x)

# --- Enhanced Patch Embedding ---
class PatchEmbedding3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config["num_channels"]
        hidden_dim = config["hidden_size"]
        channels = config["conv_channels"]
        
        self.stages = nn.ModuleList([
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[2]),
            ConvBlock(channels[2], channels[3]),
            ConvBlock(channels[3], channels[4]),
            ConvBlock(channels[4], channels[5]),
            ConvBlock(channels[5], hidden_dim),
        ])

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features[-5:]  # return last 5 feature maps for BiFPN

# --- Fixed BiFPN Block ---
class BiFPNBlock3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels_list = config["bifpn_channels"]
        out_channels = config["hidden_size"]
        
        self.projections = nn.ModuleList([
            nn.Conv3d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.fusions = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        x = [proj(f) for proj, f in zip(self.projections, features)]
        
        # Top-down pathway
        for i in range(len(x) - 2, -1, -1):
            upsampled = F.interpolate(x[i+1], size=x[i].shape[2:], mode='trilinear', align_corners=False)
            x[i] = x[i] + upsampled
        
        # Bottom-up pathway - FIXED: Use adaptive pooling to match sizes
        for i in range(1, len(x)):
            pooled = F.adaptive_avg_pool3d(x[i-1], x[i].shape[2:])
            x[i] = x[i] + pooled
        
        # Fuse features
        fused = [fusion(f) for fusion, f in zip(self.fusions, x)]
        return fused[-1]

# --- Fast Multi-Head Attention with Flash Attention ---
class FasterMultiHeadAttention(nn.Module):
    """
    Optimized multi-head attention with merged QKV projections and Flash Attention support.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config.get("qkv_bias", True)
        
        # Single linear layer for query, key, and value projections
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size * 3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        
        # Output projection
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        batch_size, sequence_length, _ = x.size()
        
        # Project query, key, and value in one operation
        qkv = self.qkv_projection(x)
        
        # Split into query, key, and value
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape to (batch_size, num_heads, seq_length, head_size)
        query = query.view(batch_size, sequence_length, self.num_attention_heads, 
                          self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, 
                      self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, 
                          self.attention_head_size).transpose(1, 2)
        
        # Flash Attention drop-in replacement
        if hasattr(F, 'scaled_dot_product_attention') and not output_attentions:
            # Use PyTorch 2.0+ Flash Attention (automatic memory optimization)
            attention_output = F.scaled_dot_product_attention(
                query, key, value, 
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
            attention_probs = None
        else:
            # Fallback to standard attention for compatibility or when attention weights needed
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attn_dropout(attention_probs)
            attention_output = torch.matmul(attention_probs, value)
        
        # Reshape back to (batch_size, seq_length, hidden_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.all_head_size)
        
        # Final projection
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        
        return (attention_output, attention_probs)

# --- Enhanced Transformer Block ---
class ImprovedTransformerBlock(nn.Module):
    def __init__(self, config, drop_path=0.0):
        super().__init__()
        self.config = config
        hidden_size = config["hidden_size"]
        mlp_dim = config["intermediate_size"]
        dropout = config["hidden_dropout_prob"]
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = FasterMultiHeadAttention(config)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Improved MLP with NewGELU
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            NewGELUActivation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, output_attentions=False):
        # Self-attention with residual connection and drop path
        attn_output, attention_probs = self.attn(self.norm1(x), output_attentions=output_attentions)
        x = x + self.drop_path(attn_output)
        
        # Feed-forward with residual connection and drop path
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return (x, attention_probs)

# --- Main Enhanced Unified 3D ViT Model ---
class BiFPN3DViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Validate configuration
        validate_config(config)
        self.config = config
        
        # Extract configuration parameters
        self.img_size = config["image_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.num_layers = config["num_hidden_layers"]
        
        # Enhanced components
        self.embedding = PatchEmbedding3D(config)
        self.bifpn = BiFPNBlock3D(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        
        # Calculate number of patches dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, self.num_channels, self.img_size, self.img_size, self.img_size)
            dummy_features = self.embedding(dummy)
            dummy_fused = self.bifpn(dummy_features)
            num_patches = rearrange(dummy_fused, 'b c d h w -> b (d h w) c').shape[1]
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.hidden_size))
        self.pos_drop = nn.Dropout(p=config["hidden_dropout_prob"])
        
        # Enhanced transformer with stochastic depth
        stochastic_depth = config.get("stochastic_depth", 0.0)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, self.num_layers)]
        self.transformer = nn.ModuleList([
            ImprovedTransformerBlock(config, drop_path=dpr[i])
            for i in range(self.num_layers)
        ])
        
        self.norm = nn.LayerNorm(self.hidden_size)
        self.attn_pool = nn.Linear(self.hidden_size, 1)
        
        # Enhanced classifier
        classifier_hidden = config.get("classifier_hidden", config["intermediate_size"])
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, classifier_hidden),
            NewGELUActivation(),
            nn.Dropout(config["hidden_dropout_prob"]),
            nn.Linear(classifier_hidden, self.num_classes)
        )
        
        # Focal loss parameters
        focal_alpha = config.get("focal_alpha", 1.0)
        focal_gamma = config.get("focal_gamma", 2.0)
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        # Apply proper weight initialization
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Input validation
        validate_input_tensor(x, self.num_channels, self.img_size)
        
        # Extract and fuse features
        feats = self.embedding(x)
        fused = self.bifpn(feats)
        x = rearrange(fused, 'b c d h w -> b (d h w) c')
        
        # Add CLS token and positional embeddings
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed[:, :x.size(1)])
        
        # Process through enhanced transformer
        all_attentions = []
        for transformer_block in self.transformer:
            x, attention_probs = transformer_block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        
        x = self.norm(x)
        
        # Dual representation: CLS token + attention pooling
        cls_out = x[:, 0]
        patch_tokens = x[:, 1:]
        weights = torch.softmax(self.attn_pool(patch_tokens), dim=1)
        patch_summary = torch.sum(weights * patch_tokens, dim=1)
        
        # Combine representations and classify
        combined = torch.cat([cls_out, patch_summary], dim=-1)
        logits = self.classifier(combined)
        
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)

    def compute_loss(self, logits, targets):
        """Compute the focal loss for handling class imbalance."""
        return self.criterion(logits, targets)
    
    def _init_weights(self, module):
        """Proper weight initialization for better convergence and stability."""
        initializer_range = self.config.get("initializer_range", 0.02)
        
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm3d):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        
        # Special initialization for embeddings
        if hasattr(module, 'pos_embed') and module.pos_embed is not None:
            torch.nn.init.trunc_normal_(module.pos_embed, mean=0.0, std=initializer_range)
        if hasattr(module, 'cls_token') and module.cls_token is not None:
            torch.nn.init.trunc_normal_(module.cls_token, mean=0.0, std=initializer_range)

def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model):
    """Print model architecture and parameter count."""
    print(f"Enhanced BiFPN3DViT Model Summary")
    print(f"=" * 50)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print(f"Model configuration: {model.config}")


def create_model(config, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BiFPN3DViT(config)
    model.to(device)

    # Print model summary
    print_model_summary(model)

    # Test forward pass
    print(f"\n" + "="*50)
    print("Testing Forward Pass")
    print("="*50)

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 128, 128, 128).to(device)
    sample_labels = torch.tensor([0, 1]).to(device)

    # Test without attention output
    logits, _ = model(input_tensor, output_attentions=False)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {logits.shape}")

    # Test loss computation
    loss = model.compute_loss(logits, sample_labels)
    print(f"Loss value: {loss.item():.4f}")

    # Test with attention output for interpretability
    logits_with_attn, all_attentions = model(input_tensor, output_attentions=True)
    print(f"Number of attention layers: {len(all_attentions) if all_attentions else 0}")
    if all_attentions:
        print(f"Attention shape per layer: {all_attentions[0].shape}")

    """Factory function to create the BiFPN3DViT model."""
    return model

