# transfer_model.py
import paramiko
import os
import logging

# SSH Configuration
SSH_HOST = 'gpu-pc-09'
SSH_USERNAME = 'bl70'
SSH_PASSWORD = 'Playstation3!'  
REMOTE_MODEL_SAVE_DIR = '/data/bl70/models/BatchTesting'

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