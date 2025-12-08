"""Custom Keras callbacks for training monitoring and control."""
import keras

from src.utils.tools import load_config


def load_callbacks_from_config(logs_dir: str, checkpoint_dir: str, steps_per_epoch: int = None) -> list:
    """
    Factory function to create callbacks from YAML config.
    
    Args:
        config_path: Path to model config YAML
        checkpoint_dir: Directory for saving checkpoints
        
    Returns:
        List of Keras callbacks
    """
    config = load_config('configs/config.yaml')['callbacks']
    callbacks = []

    # Backup every n-epochs
    if config['epoch_checkpoint']['enabled'] is True:
        save_every_epochs = config['epoch_checkpoint']['save_every']
        if steps_per_epoch:
            save_freq = int(save_every_epochs * steps_per_epoch)
        else:
            save_freq = 'epoch'

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=f'{checkpoint_dir}/epoch_{{epoch:02d}}.keras',
                save_freq=save_freq,
                verbose=config['epoch_checkpoint']['verbose']
            )
        )

    # Best model
    if config['best_checkpoint']['enabled'] is True:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=f'{checkpoint_dir}/{config["best_checkpoint"]["filename"]}',
                monitor=config['best_checkpoint']['monitor'],
                save_best_only=True,
                mode=config['best_checkpoint']['mode'],
                verbose=config['best_checkpoint']['verbose']
            )
        )

    # Early Stopping
    if config['early_stopping']['enabled'] is True:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=config['early_stopping']['monitor'],
                patience=config['early_stopping']['patience'],
                restore_best_weights=config['early_stopping']['restore_best_weights'],
                mode=config['early_stopping']['mode'],
                min_delta=config['early_stopping']['min_delta'],
                verbose=config['early_stopping']['verbose']
            )
        )

    # TensorBoard
    if config['tensorboard']['enabled']:
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=logs_dir,
                histogram_freq=config['tensorboard'].get('histogram_freq', 0),
                write_graph=config['tensorboard'].get('write_graph', True),
                write_images=config['tensorboard'].get('write_images', False)
            )
        )

    return callbacks
