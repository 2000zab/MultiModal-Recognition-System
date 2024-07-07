'''
utils_log.py

这个文件包含了一些用于日志记录的工具函数
'''
import logging
import os
import sys
import time

def init_logger(who, log_level=logging.INFO):
    """
    初始化一个logger
    
    通过who参数指定logger的名字，通过log_level参数指定logger的日志级别
    日志文件会被存储在项目主文件夹下的logs文件夹中
    日志文件的命名格式为{who}-{date}.log
    日志文件的格式为{time} - {name} - {level} - {message}
    日志会同时输出到控制台和文件中
    """
    # 创建一个logger
    logger = logging.getLogger(who)
    logger.setLevel(log_level)
    # 创建一个handler，用于写入日志文件
    # 指定录入项目主文件夹下的logs文件夹，如果没有则创建
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(main_path + '/logs'):
        os.makedirs(main_path + '/logs')
    log_file = main_path + f'/logs/{who}-{time.strftime("%Y%m%d-%H%M%S", time.localtime())}.log'
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(handler)
    # 创建一个控制台输出handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger