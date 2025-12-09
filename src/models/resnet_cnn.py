"""
Functions for building ResNet CNN.
"""
from typing import Tuple

from keras.initializers import HeNormal
from keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    Multiply,
    ReLU,
)
from keras.models import Model
from keras.regularizers import l2
import tensorflow as tf


def se_block(input_tensor: tf.Tensor, l2_value: float = 1e-4, ratio: int = 16) -> tf.Tensor:
    """
    Creates a Squeeze-and-Excitation (SE) block.

    The SE block explicitly models channel interdependencies to adaptively
    recalibrate channel-wise feature responses. It consists of a 'Squeeze' operation
    (Global Average Pooling) to capture global spatial context, followed by an
    'Excitation' operation (two Dense layers forming a bottleneck) to predict
    channel importance weights.

    This implementation includes L2 regularization on the Dense layers kernels
    and disables bias to maintain better weight control.

    Args:
        input_tensor (tf.Tensor): Input Keras tensor with shape (batch, height, width, channels).
        l2_value (float, optional): L2 regularization factor applied to the Dense layers' kernels.
            Defaults to 1e-4.
        ratio (int, optional): Reduction ratio for the bottleneck in the Excitation step.
            A higher ratio reduces the number of parameters in the SE block. Defaults to 16.

    Returns:
        tf.Tensor: Output tensor of the same shape as `input_tensor`, rescaled by the
            learned channel importance weights.

    References:
        - Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks".
    """
    filters = input_tensor.shape[-1]

    # Squeeze: Global signal averaging from whole board (1x1 pixel per channel)
    se = GlobalAveragePooling2D()(input_tensor)

    # Excitation: Two Dense layers learning correlations between channels
    # WARNING: Decided to add little bit of L2 reg, to "control" weights
    se = Dense(filters // ratio, activation='relu', use_bias=False,
                      kernel_regularizer=l2(l2_value))(se)

    se = Dense(filters, activation='sigmoid', use_bias=False,
                      kernel_regularizer=l2(l2_value))(se)

    # Input rescaling by importance weights
    return Multiply()([input_tensor, se])


def res_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int = 3,
    l2_value: float = 1e-4,
    use_bn: bool = True,
    use_se: bool = False
) -> tf.Tensor:
    """
    Creates a Residual Block (ResBlock).

    This block implements the standard ResNet architecture component consisting of:
    Convolution -> BatchNormalization (opt) -> ReLU -> Convolution -> BatchNormalization (opt).
    It features a skip connection (shortcut) that adds the original input to the output
    of the convolutional path, which helps mitigate the vanishing gradient problem.

    Optionally, it can integrate a Squeeze-and-Excitation (SE) block for channel-wise
    attention.

    Args:
        x (tf.Tensor): Input tensor.
        filters (int): Number of filters (channels) for the convolutional layers.
                       Note: Should match input channels for the Add operation to work
                       without a projection layer.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.
        l2_value (float, optional): L2 regularization factor. Defaults to 1e-4.
        use_bn (bool, optional): Whether to use Batch Normalization. Defaults to True.
        use_se (bool, optional): Whether to apply Squeeze-and-Excitation block.
                                 Defaults to False.

    Returns:
        tf.Tensor: Output tensor of the Residual Block.
    """
    init = HeNormal()

    # Saving shortcut to add at the end (skip connection)
    shortcut = x

    # --- Conv 1 ---
    y = Conv2D(
        filters,
        kernel_size,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(l2_value)
    )(x)

    if use_bn:
        y = BatchNormalization()(y)
    y = ReLU()(y)

    # --- Conv 2 ---
    y = Conv2D(
        filters,
        kernel_size,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(l2_value)
    )(y)

    if use_bn:
        y = BatchNormalization()(y)

    # --- Squeeze and Excitation ---
    if use_se:
        y = se_block(y, l2_value=l2_value)

    # Adding the shortcut (Skip Connection)
    y = Add()([shortcut, y])
    y = ReLU()(y)
    return y


def build_resnet_cnn(
    input_shape: Tuple[int, int, int] = (8, 8, 12),
    filters: int = 32,
    dense_shape: int = 256,
    n_res_blocks: int = 6,
    l2_value: float = 1e-4,
    use_batchnorm: bool = True,
    use_squeeze_and_excitation: bool = False,
    dropout_final: float = 0.4
) -> Model:
    """
    Builds a Residual Neural Network (ResNet) with a Value Head.

    This architecture follows the AlphaZero-style design patterns:
    1. Initial Convolutional block.
    2. A stack of Residual Blocks (main "body" of the network).
    3. A specific "Value Head" designed to evaluate the board state.

    The Value Head reduces dimensionality using a 1x1 convolution, flattens the output,
    passes it through a dense layer, and finally outputs a scalar value via 'tanh' activation
    (range [-1, 1]), representing the evaluation of the position (e.g., win probability or score).

    Args:
        input_shape (Tuple[int, int, int], optional): Shape of the input tensor (H, W, C).
            Defaults to (8, 8, 12).
        filters (int, optional): Number of filters in the convolutional blocks. Defaults to 32.
        dense_shape (int, optional): Size of the dense layer in the value head. Defaults to 256.
        n_res_blocks (int, optional): Number of residual blocks to stack. Defaults to 6.
        l2_value (float, optional): L2 regularization factor. Defaults to 1e-4.
        use_batchnorm (bool, optional): Whether to use Batch Normalization throughout the network.
            Defaults to True.
        use_squeeze_and_excitation (bool, optional): Whether to include SE blocks inside
            Residual Blocks. Defaults to False.
        dropout_final (float, optional): Dropout rate applied before the final output layer.
            Defaults to 0.4.

    Returns:
        keras.models.Model: A compiled Keras Model instance ready for training.
    """
    init = HeNormal()
    inp = Input(shape=input_shape, name='input')

    # --- Initial Convolution Block ---
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(l2_value),
        name='initial_conv'
    )(inp)

    if use_batchnorm:
        x = BatchNormalization(name='initial_bn')(x)
    x = ReLU(name='initial_relu')(x)

    # --- Residual Tower ---
    for _ in range(n_res_blocks):
        x = res_block(
            x,
            filters=filters,
            kernel_size=3,
            l2_value=l2_value,
            use_bn=use_batchnorm,
            use_se=use_squeeze_and_excitation
        )

    # --- Value Head ---
    # 1x1 conv to reduce depth before flattening (computationally efficient)
    v = Conv2D(
        filters=32,
        kernel_size=1,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=l2(l2_value),
        name='value_head_conv'
    )(x)

    if use_batchnorm:
        v = BatchNormalization(name='value_head_bn')(v)
    v = ReLU(name='value_head_relu')(v)

    v = Flatten(name='value_head_flatten')(v)

    v = Dense(
        dense_shape,
        kernel_initializer=init,
        kernel_regularizer=l2(l2_value),
        activation='relu',
        name='value_head_dense'
    )(v)

    if dropout_final > 0:
        v = Dropout(dropout_final, name='value_head_dropout')(v)

    # Output: Tanh for range [-1, 1]
    out = Dense(
        1,
        activation='tanh',
        name='value_output',
        kernel_initializer=init,
        kernel_regularizer=l2(l2_value)
    )(v)

    return Model(inputs=inp, outputs=out, name='resnet_value_model')
