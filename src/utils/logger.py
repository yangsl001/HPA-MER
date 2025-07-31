# src/utils/logger.py
import logging
import os
import sys


def setup_logger(log_dir, name='train_log'):
    """
    设置一个既能向控制台输出，又能写入文件的logger。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 防止重复添加handler
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件handler
    log_file = os.path.join(log_dir, f'{name}.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    # 创建控制台handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    # 创建formatter并添加到handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
# Custom logger to save console output to a file
