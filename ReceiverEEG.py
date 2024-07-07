'''
EEGReceiver类，用于接收脑电数据，存储脑电数据，存储trigger信息
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pylsl import StreamInlet, resolve_streams
import numpy as np
import time
import joblib
from collections import deque
from multiprocessing import Process
import json
import socket
import threading

from utils import init_logger
from Configs import EEGReceiver_Configs as config
from Configs import LiveAmp as device

class ReceiverEEG(Process):
    def __init__(self):
        super(ReceiverEEG, self).__init__()
        # 网络参数
        self.network_config = config.network_config
        # 其他参数
        self.config = config
        self.logger = init_logger('ReceiverEEG')
        # 内部数据
        self.block_data = deque(maxlen=30 * 60 * device.sfreq)   # 30分钟数据
        self.sample_timests = deque(maxlen=30 * 60 * device.sfreq)   # 30分钟时间戳
        self.triggers = []        # 用于存储trigger的缓存
        # 两个标志位，标识数据操作
        self.recv_flag = True       # 采集标志位，默认开启，表示lsl流开启
        self.save_flag = False      # 保存标志位，默认关闭，表示默认不保存
        self.logger.debug("ReceiverEEG initialized!")
        
    def run(self):
        '''启动接收器'''
        self.logger.debug("Receiver Started!")
        self.recv_flag = True
        self.recv_thread = threading.Thread(target=self.receiving, daemon=True)
        self.recv_trigger_thread = threading.Thread(target=self.trigger_thread, daemon=True)
        # self.recv_thread = Process(target=self.receiving, daemon=True)
        # self.recv_trigger_thread = Process(target=self.trigger_thread, daemon=True)
        self.recv_thread.start()
        self.recv_trigger_thread.start()
        self.recv_thread.join()
        self.recv_trigger_thread.join()
        self.logger.debug("Receiver Ended!")
        
    def receiving(self):
        '''接收数据的线程'''
        # 初始化
        # 尝试连接脑电采集设备，获取数据流eeg_stream
        may_be_info = resolve_streams(wait_time=1.0)
        for info in may_be_info:
            if info.type() == "EEG":
                eeg_stream = StreamInlet(info)
        if eeg_stream is None:
            self.logger.error("EEG stream not found!")
            raise Exception("EEG stream not found!")
        self.logger.debug("EEG stream found!")
        # 接收数据
        while self.recv_flag:
            sample, _ = eeg_stream.pull_sample()
            # 获取一份数据
            # 存储时间戳
            self.sample_timests.append(time.time())
            # 如果存储标志为True，存储数据
            if self.save_flag:
                self.block_data.append(np.append(sample, 0))
        # 结束采集
        eeg_stream.close_stream()
            
    def _store_block(self, filename):
        '''
        存储block_data，存储格式为pkl
        根据trigger时间戳，将trigger信息存储在block_data中
        '''
        T = len(self.block_data)
        C = len(self.block_data[0])
        data = np.array(self.block_data, copy=True)
        # 计算时间戳偏差
        offset = self._calculate_timestamp_offset()
        # 对每个trigger，利用二分查找在time_stamps中找到对应的时间戳位置，存储在data中
        for trig, exp_ts, local_ts in self.triggers:
            # 利用时间戳偏差，计算实验端时间戳对应的本地时间戳
            aligned_ts = exp_ts + offset
            idx = int(np.searchsorted(self.sample_timests, aligned_ts))
            data[idx, -1] = trig
        # 存储data
        storage = {
            'data': data,
            'mne_info': device.mne_info,
        }
        filename = filename + "_eeg.pkl"
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        joblib.dump(storage, filename)
        self.logger.debug("EEG Data Saved!!!")
        self.logger.debug(f"stores at: {filename}")
            
    def _calculate_timestamp_offset(self):
        '''
        计算时间戳偏差
        实验端的时间戳和本地时间戳之间的偏差主要包括两部分：
        1. 实验端和本地的时间差
        2. 网络传输的延迟
        我们目前的情况，实验端和本地在同一台机器上，因此时间差可以忽略，这里只计算网络传输的延迟
        利用所有trigger的时间戳对，计算平均偏差
        '''
        # 计算所有时间戳对的偏差
        offsets = [local_ts - exp_ts for _, exp_ts, local_ts in self.triggers]

        # 计算平均偏差
        average_offset = sum(offsets) / len(offsets)
        return average_offset

    def trigger_thread(self):
        '''接收trigger信息的线程'''
        # 接收trigger信息的socket
        trigger_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        trigger_socket.bind((self.network_config.IP, self.network_config.PORT))
        trigger_socket.listen(5)
        self.logger.debug("Trigger socket binded!")
        self.logger.debug("Trigger thread started!")
        while self.recv_flag:
            client, addr = trigger_socket.accept()
            self.logger.debug(f"Trigger Receiver Waiting for Trigger from {addr}...")
            while self.recv_flag:
                raw_json = client.recv(1024)
                local_ts = time.time()
                if not raw_json:
                    break
                trig = json.loads(raw_json)
                self._parse_trigger(trig, local_ts)
            client.close()
        self.logger.debug("Trigger thread ended!")
        trigger_socket.close()
    
    def _parse_trigger(self, trig, local_ts):
        '''解析trigger信息'''
        cmd, args = trig
        print(cmd, args, local_ts)
        if cmd == 'sess_start':
            self._sess_start(args, local_ts)
        elif cmd == 'sess_end':
            self._sess_end(args, local_ts)
        elif cmd == 'store':
            self._store(args, local_ts)
        elif cmd == 'trigger':
            self._trigger(args, local_ts)
        elif cmd == 'kill':
            self._kill(args, local_ts)
        else:
            self.logger.error(f"Trigger Command {cmd} Not Found!")
            
    # 以下是一些默认的trigger方法
    def _sess_start(self, args, local_ts):
        '''开始采集，这个方法只会在一次实验开始时调用一次，表示实验数据开始
        参数args: {
            'timestamp': 来自实验端的时间戳
        }
        操作：需要开启存储功能，清空block，记录实验端时间戳
        '''
        # 首先清空存储数据
        self.block_data.clear()
        self.triggers.clear()
        self.sample_timests.clear()
        # 然后开启存储功能
        self.save_flag = True
        self.logger.debug("Session Start!")
        
    def _sess_end(self, args, local_ts):
        '''结束采集，这个方法只会在一次实验结束时调用一次，表示实验数据结束
        参数args: {
            'timestamp': 来自实验端的时间戳
        }
        操作：关闭存储功能
        '''
        self.save_flag = False
        self.logger.debug("Session End!")
        
    def _store(self, args, local_ts):
        '''保存采集，这个方法只会在一次实验结束时调用一次，调用结束存储功能
        参数args: {
            'filename': 存储文件名
            'timestamp': 来自实验端的时间戳
        }
        操作：保存数据，存储结果
        '''
        if self.save_flag:
            self.save_flag = False
        self._store_block(args['filename'])
        self.logger.debug(f"Store at {args['filename']}")
        
    def _trigger(self, args, local_ts):
        '''存储trigger信息，这个方法会在实验过程中多次调用，表示实验过程中的触发信息
        参数args: {
            'trigger': trigger信息
            'timestamp': 来自实验端的时间戳
        }
        操作：存储trigger信息
        '''
        trig, exp_ts = args['trigger'], args['timestamp']
        self.triggers.append([trig, exp_ts, local_ts])
        self.logger.debug(f"Trigger {trig} at Local {local_ts}, Exp {exp_ts}")
        
    def _kill(self, args, local_ts):
        '''终止采集，这个方法只会在所有实验结束时调用一次，表示采集终止，程序结束
        参数args: {
            'timestamp': 来自实验端的时间戳
        }
        '''
        self.save_flag = False
        self.recv_flag = False
        self.logger.debug("Kill!")
        
if __name__ == '__main__':
    receiver = ReceiverEEG()
    receiver.start()
    receiver.join()