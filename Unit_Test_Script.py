from utils import *
from Configs import EEGReceiver_Configs as eeg_config
import logging
import os
import socket
import json
import time
import joblib

def test_init_logger():
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logger = init_logger('test', 1)
    assert logger.name == 'test'
    assert len(logger.handlers) == 2
    logger.info('test')
    logger.debug('test_debug')
    logger.error('test_error')

def test_eeg_receiver():
    time.sleep(10)
    # 首先建立一个socket连接
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((eeg_config.network_config.IP, eeg_config.network_config.PORT))
    # 发送开始信号
    client.send(json.dumps(['sess_start', {'timestamp': time.time()}]).encode())
    time.sleep(2)
    client.send(json.dumps(['trigger', {'timestamp': time.time(), 'trigger': 1}]).encode())
    time.sleep(2)
    client.send(json.dumps(['trigger', {'timestamp': time.time(), 'trigger': 2}]).encode())
    time.sleep(2)
    client.send(json.dumps(['store', {'filename': 'subject_data/eeg_test_1'}]).encode())
    client.send(json.dumps(['sess_end', {'timestamp': time.time()}]).encode())
    time.sleep(2)
    client.send(json.dumps(['sess_start', {'timestamp': time.time()}]).encode())
    time.sleep(2)
    client.send(json.dumps(['trigger', {'timestamp': time.time(), 'trigger': 1}]).encode())
    time.sleep(2)
    client.send(json.dumps(['trigger', {'timestamp': time.time(), 'trigger': 2}]).encode())
    time.sleep(2)
    client.send(json.dumps(['store', {'filename': 'subject_data/eeg_test_2'}]).encode())
    client.send(json.dumps(['sess_end', {'timestamp': time.time()}]).encode())
    time.sleep(2)
    # client.send(json.dumps(['kill', {'timestamp': time.time()}]).encode())

if __name__ == '__main__':
    test_init_logger()
    # test_eeg_receiver()
    print('All tests passed!')