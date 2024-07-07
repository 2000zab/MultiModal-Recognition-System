import time
import os
from psychopy import core, gui, event, visual, sound
event.globalKeys.add('escape', func=core.quit)
from psychopy.tools.filetools import fromFile, toFile
from random import shuffle
from Configs import EEGReceiver_Configs as eeg_config
from Configs import global_config
from Configs import GazeReceiver_Configs as gaze_config
import socket
import pandas as pd
from multiprocessing import Queue
import numpy as np
import json
from utils import init_logger

class Controller:
    def __init__(self):
        self.eeg_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.eeg_client.connect((eeg_config.network_config.IP, eeg_config.network_config.PORT))
        self.gaze_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.gaze_client.connect((gaze_config.network_config.DATA_IP, gaze_config.network_config.DATA_PORT))
        
    def sess_start(self):
        self.eeg_client.send(json.dumps(['sess_start', {'timestamp': time.time()}]).encode())
        self.gaze_client.send(json.dumps(['sess_start', {'timestamp': time.time()}]).encode())
    
    def push_trigger(self, trigger):
        self.eeg_client.send(json.dumps(['trigger', {'timestamp': time.time(), 'trigger': global_config.triggers[trigger]}]).encode())
        self.gaze_client.send(json.dumps(['trigger', {'timestamp': time.time(), 'trigger': global_config.triggers[trigger]}]).encode())
        
    def store(self, filename):
        self.eeg_client.send(json.dumps(['store', {'filename': filename}]).encode())
        self.gaze_client.send(json.dumps(['store', {'filename': filename}]).encode())
    
    def sess_end(self):
        self.eeg_client.send(json.dumps(['sess_end', {'timestamp': time.time()}]).encode())
        self.gaze_client.send(json.dumps(['sess_end', {'timestamp': time.time()}]).encode())

def play_movie(win, file, keys):
    mov = visual.MovieStim(win, file, size=(1920, 1080))
    # 按照顺序播放视频，每按一次键，视频前进5-8帧，达到30帧后正常播放
    total_frames = 90
    while total_frames >= 60:
        tmp = event.waitKeys()
        N = np.random.randint(5, 8)
        for _ in range(N):
            mov.draw()
            win.flip()
            total_frames -= 1
    for _ in range(total_frames):
        mov.draw()
        win.flip()
    win.flip()

if __name__ == '__main__':
    # ============================== 实验参数设置 ==============================
    # trigger参数设置
    triggers = global_config.triggers

    # 设置实验类别
    n_classes = 2
    classes = ['left_hand', 'right_hand']
    exp_loops = 8 # 实验循环次数

    # 文件路径参数
    save_data_dir = global_config.data_dir
    local_exp_dir = global_config.exp_dir

    # 设置后台控制器
    controller = Controller()
    
    # 设置全局事件
    def quit_cali():
        controller.sess_end()
        win.close()
        core.quit()
    event.globalKeys.add(key='escape', func=quit_cali)
    
    # ============================== 受试者信息采集 ==============================
    try:
        cache_path = os.path.join(local_exp_dir, 'lastParams.pickle')
        data_win = fromFile(cache_path)
        data_win['实验编号'] += 1
    except:
        data_win = {'受试者编号': 1, '实验编号': 1}
    dlg = gui.DlgFromDict(data_win, title='实验信息')
    if dlg.OK:
        toFile(cache_path, data_win)
    else:
        core.quit()
    subject_id = data_win['受试者编号']
    block_id = data_win['实验编号']

    # 日志设置
    logger = init_logger('Exp_Motor_Execution')
    exp_start_time = time.time()
    logger.info(f'===============================================================')
    logger.info(f'Experiment start!!! Block for EEG experiment without feedback')
    logger.info(f'Experiment start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_start_time))}')
    logger.info(f'Subject ID: {subject_id}, Block ID: {block_id}')
    logger.info(f'===============================================================')

    # ============================== 实验运行流程 ==============================
    # 2. 实验运行

    # 向后台控制器设置受试者编号
    time.sleep(5)
    
    # 0. 窗口设置
    win = visual.Window(
        size=(1920, 1080), 
        allowGUI=True,
        units='pix', 
        fullscr=True,
        color='#fdeca6'
    )
    event.Mouse(visible=False)

    # 0.1 视线校准
    # controller.starting()
    # GazeCalibrate(win, controller)

    # 清空空闲条件下脑电和面部数据采集，开始实验
    controller.sess_start()
    logger.info(f'Session start!!!')
    # 1. 实验刺激
    stims = classes * exp_loops
    shuffle(stims)
    # 实验过程中的展示内容
    beep = sound.Sound('A', secs=0.5)
    fixation = visual.ImageStim(win, "resources/fixation.png")
    class_show_dict = {
        'left_hand': visual.ImageStim(win, "resources/left_hand.png"),
        'right_hand': visual.ImageStim(win, "resources/right_hand.png"),
        'stay': visual.ImageStim(win, "resources/cube_stay.jpg"),
    }
    show_dict = {
        'left_hand': "resources/cube_left_rot.mp4",
        'right_hand': "resources/cube_right_rot.mp4",
        'no_move': visual.ImageStim(win, "resources/no_move.png"),
    }
    keys_dict = {
        'left_hand': ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c'],
        'right_hand': ['num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7', 'num_8', 'num_9'],
    }
    start = visual.ImageStim(win, "resources/start.png")
    resting = visual.ImageStim(win, "resources/exp_break.png")
    end = visual.ImageStim(win, "resources/end.png")

    # 2. 实验运行
    # 2.1 实验开始
    start.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    win.flip()

    # 2.2 实验运行
    for stim in stims:
        # 2.3 休息
        resting.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        win.flip()
        
        # 首先保持不动
        show_dict['no_move'].draw()
        win.flip()
        controller.push_trigger('no_move')
        core.wait(4)
        win.flip()
        core.wait(0.5)
        
        # 2.4 提示
        # 2.4.1 注视点+类别刺激
        class_show_dict[stim].draw()
        win.flip()
        core.wait(2)
        # 2.4.2 声音提示，先是静止的立方体
        beep.play()
        class_show_dict['stay'].draw()
        win.flip()
        # 2.4.3 触发器
        controller.push_trigger(stim)
        logger.info(f"Stimulus: {stim}!!!")
        core.wait(1)
        play_movie(win, show_dict[stim], keys_dict[stim])
        win.flip()
        core.wait(1)
        
        # 2.5 结束
        win.flip()
        logger.info(f"Stimulus end!!!")
        core.wait(2)
        
    # 2.6 实验结束
    end.draw()
    win.flip()
    core.wait(3)
    win.close()

    # ======================== 实验数据存储 ========================
    exp_end_time = time.time()
    controller.store(f'./{save_data_dir}/{subject_id:03d}/block_{block_id:03d}_Exp_Motor_Execution')
    time.sleep(5) # 等待数据存储完成
    controller.sess_end()
    exp_df = pd.DataFrame({
        'sub_id': [subject_id],
        'block_id': [block_id],
        'exp_type': ['Exp_Motor_Execution'],
        'start_time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_start_time))],
        'end_time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_end_time))],
        'eeg_filename': [f'./data/{subject_id:03d}/block_{block_id:03d}_eeg_noise.pkl'],
        'gaze_filename': [f'./data/{subject_id:03d}/block_{block_id:03d}_gaze_noise.pkl']
    })
    exp_df.to_csv(os.path.join(local_exp_dir, 'exp_records.csv'), mode='a', header=False, index=None)
    time.sleep(5)
    logger.info(f'All data saved!!!')
    logger.info(f'Experiment end!!!')
    logger.info(f"Total time: {time.time() - exp_start_time} s")
    logger.info(f'===============================================================')