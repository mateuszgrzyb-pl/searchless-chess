"""
Training ResNet - M-size.
"""
import glob
import random

from loguru import logger
import keras
import tensorflow as tf

from src.models.resnet_cnn import build_resnet_cnn
from src.utils.tools import create_dataset, load_config, setup_training_environment
from src.training.callbacks import load_callbacks_from_config

def main():
    """
    Creates and train ResNet.
    2.3M parameters.
    """
    # Logging
    model_name = 'model_02_resnet_m'
    experiment_log_dir, checkpoints_dir, tensorboard_dir = setup_training_environment(model_name)

    logger.add(experiment_log_dir / "training_{time:YYYY-MM-DD}.log",
         rotation="5 MB",
         retention=10)
    logger.info(f" {'Creating new ResNet CNN'} ".center(100, "="))
    logger.info(f" {'2.3M trainable parameters'} ".center(100, "="))

    # Step 1. Configuration
    logger.info('Step 1. Loading config file.')
    config = load_config('configs/config.yaml')
    data_path = config['data']['paths']['stage_5_unified']
    train_ratio = config['split']['train_ratio']
    val_ratio = config['split']['val_ratio']
    random_seed = config['split']['random_seed']
    batch_size = config['modelling']['batch_size']
    learning_rate = config['modelling']['learning_rate']
    clipnorm = config['modelling']['clipnorm']
    num_of_epochs = config['modelling']['epochs']

    random.seed(random_seed)
    tf.random.set_seed(random_seed)
    keras.mixed_precision.set_global_policy('mixed_float16')

    total_samples = config['split']['total_samples']
    total_samples_train = int(int(total_samples) * train_ratio)
    total_samples_val   = int(int(total_samples) * val_ratio)

    #   Full steps
    steps_per_epoch_train = total_samples_train // batch_size
    steps_per_epoch_val   = total_samples_val // batch_size

    #   Virtual steps (1/10) - to not wait hours for results
    virtual_steps_train = steps_per_epoch_train // 10
    virtual_steps_val = steps_per_epoch_val   // 10

    # Step 2. Reading datafiles
    logger.info('Step 2. Reading datafiles.')
    logger.info(f"Loading datafiles from path: {data_path}.")
    all_files = glob.glob(data_path + "/*.tfrec")
    logger.info(f"Found {len(all_files)} datafiles.")
    random.shuffle(all_files)
    split_idx = int(len(all_files) * train_ratio)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    train_dataset = create_dataset(train_files, is_training=True, batch_size=batch_size)
    val_dataset = create_dataset(val_files, is_training=False, batch_size=batch_size)

    # Step 3. Preparing callbacks
    logger.info('Step 3. Preparing callbacks.')
    callbacks = load_callbacks_from_config(logs_dir=tensorboard_dir, checkpoint_dir=checkpoints_dir, steps_per_epoch=virtual_steps_train)

    # Step 4. Preparing model
    logger.info('Step 4. Preparing model.')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    model = build_resnet_cnn(
        input_shape=(8, 8, 12),
        filters=128,
        dense_shape=256,
        n_res_blocks=6,
        l2_value=1e-4,
        use_batchnorm=True,
        use_squeeze_and_excitation=False,
        dropout_final=0.4
    )

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Step 5. Modelling
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_of_epochs,
        steps_per_epoch=virtual_steps_train,
        validation_steps=virtual_steps_val,
        callbacks=callbacks,
        verbose=1
    )
    model.summary()
    logger.success('Modelling finished!')

if __name__ == '__main__':
    main()
