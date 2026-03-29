# --- Enhanced Configuration ---
enhanced_config = {
    # Basic model parameters
    "image_size": 128,
    "num_channels": 1,
    "num_classes": 3,
    "hidden_size": 384,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 768,
    
    # Dropout and regularization
    "hidden_dropout_prob": 0.4,
    "attention_probs_dropout_prob": 0.4,
    "stochastic_depth": 0.4,
    
    # Convolutional feature extraction
    "conv_channels": [16, 32, 64, 128, 256, 512],
    "bifpn_channels": [64, 128, 256, 512, 384],
    
    # Loss function parameters
    "focal_alpha": 1.0,
    "focal_gamma": 2.0,
    
    # Training parameters
    "initializer_range": 0.02,
    "qkv_bias": True,
    "classifier_hidden": 768,
    
    "model_name": "BiFPN3DViT",
}