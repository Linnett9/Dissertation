import paramiko
import os
import logging
import json

# SSH Configuration
SSH_HOST = 'gpu-pc-09'
SSH_USERNAME = 'bl70'
SSH_PASSWORD = 'Playstation3!'  
REMOTE_MODEL_SAVE_DIR = '/data/bl70/models/InceptionResNetCondModel'
REMOTE_TUNER_RESULTS_DIR = '/data/bl70/TestingInceptionResNetCondTunerResults'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_remote_dir_exists(sftp, remote_directory):
    dirs = remote_directory.split('/')
    current_dir = ''
    for dir in dirs:
        if dir:
            current_dir += f'/{dir}'
            try:
                sftp.stat(current_dir)
                logger.info(f"Directory exists: {current_dir}")
            except FileNotFoundError:
                try:
                    sftp.mkdir(current_dir)
                    logger.info(f"Directory created: {current_dir}")
                except PermissionError as e:
                    logger.error(f"Permission denied: Cannot create directory {current_dir}. Error: {e}")
                    raise

def transfer_model_to_gpu(local_path, remote_path, old_remote_path=None):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, username=SSH_USERNAME, password=SSH_PASSWORD)
    sftp = ssh.open_sftp()
    
    remote_directory = os.path.dirname(remote_path)
    ensure_remote_dir_exists(sftp, remote_directory)
    
    try:
        logger.info(f"Transferring file from {local_path} to {remote_path}")
        sftp.put(local_path, remote_path)
        logger.info(f"File transferred successfully to {remote_path}")
        if old_remote_path:
            try:
                sftp.remove(old_remote_path)
                logger.info(f"Old model removed: {old_remote_path}")
            except FileNotFoundError:
                logger.info(f"Old model not found, skipping removal: {old_remote_path}")
    except Exception as e:
        logger.error(f"Error during file transfer: {e}")
        raise
    finally:
        sftp.close()
        ssh.close()

def transfer_tuner_results_to_gpu(local_output_dir, remote_dir=REMOTE_TUNER_RESULTS_DIR):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(SSH_HOST, username=SSH_USERNAME, password=SSH_PASSWORD)
    sftp = ssh.open_sftp()
    
    try:
        ensure_remote_dir_exists(sftp, remote_dir)
        
        # Transfer best model
        local_model_path = os.path.join(local_output_dir, 'best_model.keras')
        remote_model_path = os.path.join(remote_dir, 'best_model.keras')
        if os.path.exists(local_model_path):
            sftp.put(local_model_path, remote_model_path)
            logger.info(f"Best model transferred successfully to {remote_model_path}")
        else:
            logger.error(f"Model file not found at {local_model_path}")
        
        # Transfer best configuration
        local_config_path = os.path.join(local_output_dir, 'best_configuration.txt')
        remote_config_path = os.path.join(remote_dir, 'best_configuration.txt')
        if os.path.exists(local_config_path):
            sftp.put(local_config_path, remote_config_path)
            logger.info(f"Best configuration transferred successfully to {remote_config_path}")
        else:
            logger.error(f"Configuration file not found at {local_config_path}")
        
        # Transfer best results
        local_results_path = os.path.join(local_output_dir, 'best_results.txt')
        remote_results_path = os.path.join(remote_dir, 'best_results.txt')
        if os.path.exists(local_results_path):
            sftp.put(local_results_path, remote_results_path)
            logger.info(f"Best results transferred successfully to {remote_results_path}")
        else:
            logger.error(f"Results file not found at {local_results_path}")
        
    except Exception as e:
        logger.error(f"Error during tuner results transfer: {e}")
        raise
    finally:
        sftp.close()
        ssh.close()



def save_best_configuration(best_config, best_val_accuracy, output_dir):
    # Save the best configuration
    with open(os.path.join(output_dir, 'best_configuration.txt'), 'w') as f:
        f.write(f'Best validation accuracy: {best_val_accuracy * 100:.2f}%\n')
        f.write(f'Best hyperparameters:\n')
        for key, value in best_config.values.items():
            f.write(f'{key}: {value}\n')


def save_best_results(best_val_accuracy, best_config, output_dir):
    best_config_file = os.path.join(output_dir, 'best_config.txt')
    with open(best_config_file, 'w') as f:
        f.write(f"Best validation accuracy: {best_val_accuracy}\n")
        f.write(f"Best hyperparameters:\n")
        for key, value in best_config.values.items():
            f.write(f"{key}: {value}\n")