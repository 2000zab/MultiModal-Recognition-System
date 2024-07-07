from psychopy import visual, core, event, gui, sound
event.globalKeys.add('escape', func=core.quit)
from psychopy.tools.filetools import fromFile, toFile
import os
import socket
import time
import json
from Configs import EEGReceiver_Configs as eeg_config
from Configs import global_config
from Configs import GazeReceiver_Configs as gaze_config
from utils import init_logger
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd

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

if __name__ == '__main__':
    # 文件路径参数
    local_data_dir = global_config.data_dir
    local_exp_dir = global_config.exp_dir

    # 设置通信socket
    controller = Controller()
    
    # 设置全局事件
    def quit_cali():
        controller.sess_end()
        win.close()
        core.quit()
    event.globalKeys.add(key='escape', func=quit_cali)
    
    # 实验信息获取
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
    logger = init_logger('Exp_Noise')
    exp_start_time = time.time()
    logger.info(f'===============================================================')
    logger.info(f'Experiment start!!! Block for noise experiment')
    logger.info(f'Experiment start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_start_time))}')
    logger.info(f'Subject ID: {subject_id}, Block ID: {block_id}')
    logger.info(f'===============================================================')

    # 0. 窗口设置
    win = visual.Window([1920, 1080], 
                        allowGUI=True, 
                        fullscr=True, 
                        units='pix', 
                        color='#fdeca6')
    event.Mouse(visible=False)
    
    # 0.1 视线校准
    controller.sess_start()
    logger.info(f'Session start!!!')

    # 1. 实验刺激设置
    beep = sound.Sound('A', secs=0.5, hamming=True)
    # 一次实验的流程如下
    def noise_stim_play(stim, stim_time):
        # 给出stim对应的提示
        instruction = visual.ImageStim(win, image=f'resources/{stim}.png')
        during = visual.ImageStim(win, image=f'resources/{stim}_during.png')
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        win.flip()
        
        # 等待1s后给出提示音
        core.wait(2)
        during.draw()
        nextFlip = win.getFutureFlipTime(clock='ptb')
        beep.play(when=nextFlip)
        win.flip()
        controller.push_trigger(stim)
        logger.info(f'noise_stim_alert: {stim}!!!')
        core.wait(stim_time)
        
        # 等待time时长后给出结束提示
        win.flip()
        logger.info(f'noise_stim_end!!!')
        core.wait(5)
        
    # 1.0 实验开始提示
    start = visual.ImageStim(win, image='resources/start.png')
    start.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    # 1.1 眨眼5s*2
    noise_stim_play('blink', 5)
    # noise_stim_play('blink', 5)

    # 1.2 眼球上下移动5s*2
    noise_stim_play('eyeball_updown', 5)
    # noise_stim_play('eyeball_updown', 5)

    # 1.3 眼球左右移动5s*2
    # noise_stim_play('eyeball_leftright', 5)
    noise_stim_play('eyeball_leftright', 5)

    # 1.4 头部转动5s*2
    # noise_stim_play('head_move', 5)
    noise_stim_play('head_move', 5)

    # 1.5 咬紧下巴5s*2
    # noise_stim_play('jaw_clench', 5)
    noise_stim_play('jaw_clench', 5)

    # 1.6 静息30s
    noise_stim_play('rest', 30)
    beep.play()

    # 结束实验
    end = visual.ImageStim(win, image='resources/end.png')
    end.draw()
    win.flip()
    core.wait(3)
    win.close()

    # ==================================================
    exp_end_time = time.time()
    # 保存数据
    controller.store(f'./{local_data_dir}/{subject_id:03d}/block_{block_id:03d}_noise')

    # 记录实验
    exp_df = pd.DataFrame({
        'sub_id': [subject_id],
        'block_id': [block_id],
        'exp_type': ['noise'],
        'start_time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_start_time))],
        'end_time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp_end_time))],
        'eeg_filename': [f'./data/{subject_id:03d}/block_{block_id:03d}_eeg_noise.pkl'],
        'gaze_filename': [f'./data/{subject_id:03d}/block_{block_id:03d}_gaze_noise.pkl']
    })
    exp_df.to_csv(os.path.join(local_exp_dir, 'exp_records.csv'), mode='a', header=False, index=None)
    time.sleep(5)
    controller.sess_end()
    logger.info(f'Experiment end!!!')
    logger.info(f'===============================================================')