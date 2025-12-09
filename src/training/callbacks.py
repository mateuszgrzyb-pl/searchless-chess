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
    backup_cfg = config.get('epoch_checkpoint', {})
    if backup_cfg.get('enabled', False) is True:
        save_every_epochs = backup_cfg.get('save_every', 5)
        if steps_per_epoch:
            save_freq = int(save_every_epochs * steps_per_epoch)
        else:
            save_freq = 'epoch'

        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=f'{checkpoint_dir}/epoch_{{epoch:02d}}.keras',
                save_freq=save_freq,
                verbose=backup_cfg.get('verbose', 0)
            )
        )

    # Best model
    best_cfg = config.get('best_checkpoint', {})
    if best_cfg.get('enabled', False) is True:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=f'{checkpoint_dir}/{best_cfg.get("filename", "best_model.keras")}',
                monitor=best_cfg.get('monitor', 'val_loss'),
                save_best_only=True,
                mode=best_cfg.get('mode', 'min'),
                verbose=best_cfg.get('verbose', 0)
            )
        )

    # Early Stopping
    early_stop_cfg = config.get('early_stopping', {})
    if early_stop_cfg.get('enabled', False) is True:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=early_stop_cfg.get('monitor', 'val_loss'),
                patience=early_stop_cfg.get('patience', 5),
                restore_best_weights=early_stop_cfg.get('restore_best_weights', True),
                mode=early_stop_cfg.get('mode', 'min'),
                min_delta=early_stop_cfg.get('min_delta', 0.00005),
                verbose=early_stop_cfg.get('verbose', 0)
            )
        )

    # Reduce LR on Plateau
    reduce_lr_cfg = config.get('reduce_lr_on_plateau', {})
    if reduce_lr_cfg.get('enabled', False):
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=reduce_lr_cfg.get('monitor', 'val_loss'),
                factor=reduce_lr_cfg.get('factor', 0.1),
                patience=reduce_lr_cfg.get('patience', 10),
                min_lr=reduce_lr_cfg.get('min_lr', 0.0),
                verbose=reduce_lr_cfg.get('verbose', 0)
            )
        )

    # TensorBoard
    tensorboard_cfg = config.get('tensorboard', {})
    if tensorboard_cfg.get('enabled', False):
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=logs_dir,
                histogram_freq=tensorboard_cfg.get('histogram_freq', 0),
                write_graph=tensorboard_cfg.get('write_graph', True),
                write_images=tensorboard_cfg.get('write_images', False)
            )
        )

    return callbacks
