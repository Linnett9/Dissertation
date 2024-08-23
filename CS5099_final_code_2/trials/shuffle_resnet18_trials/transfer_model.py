import paramiko
import os
import logging
import json

# SSH Configuration
SSH_HOST = 'gpu-pc-09'
SSH_USERNAME = 'bl70'
SSH_PASSWORD = 'Playstation3!'
REMOTE_MODEL_SAVE_DIR = '/data/bl70/models/ShuffleResNetCondPotentialFinal2'
REMOTE_TUNER_RESULTS_DIR = '/data/bl70/ShuffleResNetPotentialFinal2'

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
        
        # Transfer the entire directory
        for root, dirs, files in os.walk(local_output_dir):
            for directory in dirs:
                local_path = os.path.join(root, directory)
                remote_path = os.path.join(remote_dir, os.path.relpath(local_path, local_output_dir))
                try:
                    sftp.mkdir(remote_path)
                except IOError:
                    pass  # Directory might already exist
            
            for file in files:
                local_path = os.path.join(root, file)
                remote_path = os.path.join(remote_dir, os.path.relpath(local_path, local_output_dir))
                sftp.put(local_path, remote_path)
                logger.info(f"Transferred {local_path} to {remote_path}")
        
        logger.info(f"All tuner results transferred successfully to {remote_dir}")
        
    except Exception as e:
        logger.error(f"Error during tuner results transfer: {e}")
        raise
    finally:
        sftp.close()
        ssh.close()

# Helper functions to save results locally
def save_best_configuration(best_config, best_val_accuracy, output_dir):
    config_path = os.path.join(output_dir, 'best_configuration.json')
    with open(config_path, 'w') as f:
        json.dump({"best_config": best_config, "best_val_accuracy": best_val_accuracy}, f, indent=2)
    logger.info(f"Best configuration saved to {config_path}")

def save_best_results(best_val_accuracy, best_config, output_dir):
    results_path = os.path.join(output_dir, 'best_results.json')
    with open(results_path, 'w') as f:
        json.dump({"best_val_accuracy": best_val_accuracy, "best_config": best_config}, f, indent=2)
    logger.info(f"Best results saved to {results_path}")