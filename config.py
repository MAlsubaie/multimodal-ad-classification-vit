# --- Biomarker-Conditioned ViT Configuration ---
enhanced_config = {
    # Basic model parameters
    "image_size": 128,
    "patch_size": 16,
    "num_channels": 1,
    "num_classes": 3,
    "num_biomarkers": 5,
    
    # Transformer parameters
    "hidden_size": 768,          # embed_dim in the model
    "num_hidden_layers": 12,     # depth
    "num_attention_heads": 12,
    "mlp_ratio": 4.0,
    
    # Regularization
    "dropout": 0.1,
    
    # Biomarker Conditioning Layers
    "conditioning_layers": [4, 5, 6, 7, 8, 9, 10, 11],
    
    "model_name": "BiomarkerConditionedViT",
}