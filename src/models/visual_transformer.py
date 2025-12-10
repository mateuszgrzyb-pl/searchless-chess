"""
Functions for building Visual Transformer.
"""
from typing import Tuple

from keras.layers import (
    Add,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    LayerNormalization,
    MultiHeadAttention,
    Reshape
)
from keras.models import Model
import tensorflow as tf


def build_vit(
    input_shape: Tuple[int, int, int] = (8, 8, 12),
    projection_dim: int = 384,
    num_heads: int = 6,
    transformer_layers: int = 8,
    dropout_rate: float = 0.1
) -> Model:
    """
    Builds a functional Vision Transformer (ViT) model.

    This implementation uses a patch-flattening embedding, learned positional
    embeddings, multiple Transformer encoder blocks (pre-norm architecture),
    and a global average pooling head for final representation aggregation.
    The model outputs a single scalar value in the range (-1, 1)
    using a tanh activation.

    Parameters
    ----------
    input_shape : tuple of int, default (8, 8, 12)
        Shape of the input tensor (height, width, channels).
        The number of patches is computed as height * width.
    projection_dim : int, default 384
        Dimensionality of the patch embeddings and Transformer hidden size.
    num_heads : int, default 6
        Number of attention heads in each Multi-Head Attention layer.
    transformer_layers : int, default 8
        Number of Transformer encoder blocks to apply.
    dropout_rate : float, default 0.1
        Dropout rate used in attention and feed-forward sublayers.

    Returns
    -------
    keras.Model
        A compiled Vision Transformer model with the specified configuration.

    Notes
    -----
    - Uses global average pooling instead of a CLS token.
    - Feed-forward hidden size is set to 2 × projection_dim.
    - Positional embeddings are correctly broadcast across the batch dimension.
    """
    inputs = Input(shape=input_shape)

    # --- 1. Patching ---
    num_patches = input_shape[0] * input_shape[1]
    patches = Reshape((num_patches, input_shape[2]))(inputs)

    # --- 2. Encoding ---
    encoded_patches = Dense(projection_dim)(patches)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding_layer = Embedding(input_dim=num_patches, output_dim=projection_dim)
    pos_embeddings = position_embedding_layer(positions)[tf.newaxis, ...]
    encoded_patches = encoded_patches + pos_embeddings

    # --- 3. Transformer Blocks ---
    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        head_dim = projection_dim // num_heads
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_dim,
            dropout=dropout_rate
        )(x1, x1)

        x2 = Add()([attention_output, encoded_patches])

        # Feed Forward
        x3 = LayerNormalization(epsilon=1e-6)(x2)  # maybe 4x would be better? Or maybe it's big enough?
        x3 = Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = Dropout(dropout_rate)(x3)
        x3 = Dense(projection_dim)(x3)

        encoded_patches = Add()([x3, x2])

    # --- 4. Head ---
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = GlobalAveragePooling1D()(representation)

    x = Dropout(dropout_rate)(representation)

    outputs = Dense(1, activation="tanh", dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name="vit_value_model")
    return model
