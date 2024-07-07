from easydict import EasyDict as edict

EEGReceiver_Configs = edict()
EEGReceiver_Configs.network_config = edict()
EEGReceiver_Configs.network_config.IP = '127.0.0.1'
EEGReceiver_Configs.network_config.PORT = 12345

GazeReceiver_Configs = edict()
GazeReceiver_Configs.network_config = edict()
GazeReceiver_Configs.network_config.DATA_IP = '127.0.0.1'
GazeReceiver_Configs.network_config.DATA_PORT = 20020
GazeReceiver_Configs.network_config.TRIGGER_IP = '127.0.0.1'
GazeReceiver_Configs.network_config.TRIGGER_PORT = 12346

LiteFeedback_Configs = edict()
LiteFeedback_Configs.network_config = edict()
LiteFeedback_Configs.network_config.IP = '127.0.0.1'
LiteFeedback_Configs.network_config.PORT = 12347

LiveAmp = edict()
LiveAmp.identifier = 'LiveAmp'
LiveAmp.sfreq = 500
LiveAmp.n_channels = 32
LiveAmp.mne_info = {
    'sfreq': 500,
    'ch_names': ['Fp1', 'Fp2', 'F3',  'F4',  'C3', 
                 'C4',  'P3',  'P4',  'O1',  'O2', 
                 'F7',  'F8',  'T7',  'T8',  'P7', 
                 'P8',  'Fz',  'Cz',  'Pz',  'FC1', 
                 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 
                 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 
                 'ECG', 'GSR'],
    'ch_types': ['eeg'] * 30 + ['ecg', 'ecg'],
}
LiveAmp.Description = '''
LiveAmp 32 channels EEG system
Serial number: 00000000
Sampling frequency: 500Hz
Using 32-channels EEG cap，with wet electrodes
Cap layout: 10-20 system
Cap Serial number: 00000000
'''

GazeCamera = edict()
GazeCamera.identifier = 'GazeCamera'
GazeCamera.sfreq = 30
GazeCamera.n_feats = 9
GazeCamera.identifier = 'logi Camera'
GazeCamera.Description = '''
Logitech BRIO 500 webcam
'''

global_config = edict()
# 文件路径配置
global_config.data_dir = "subject_data"             # 数据存放路径
global_config.model_dir = "models"          # 模型存放路径
global_config.exp_dir = "records"           # 实验记录存放路径
global_config.log_dir = "logs"              # 日志存放路径
global_config.cali_data_path = "records/last_cali_data.pkl"
# trigger设置
global_config.triggers = {                  # trigger设置
    'blink': 150,                               # 眨眼
    'eyeball_ud': 151,                          # 眼球上下移动
    'eyeball_lr': 152,                          # 眼球左右移动
    'head_move': 153,                           # 头部移动
    'jaw_clench': 154,                          # 咬牙
    'rest_state': 155,                          # 静息状态
    'left_hand': 200,                           # 左手
    'right_hand': 201,                          # 右手
    'no_move': 202,                             # 保持不动
    'both_hand': 203,                           # 双手
    'feet': 204,                                # 双脚
}
global_config.unity_cmd = {                 # unity控制命令
    'LEFT',
    'RIGHT',
}
# 反馈设置
global_config.predict_time = 6.8             # 预测时间
global_config.cut_time = 2                   # 截取时间
global_config.max_blocks = 5                 # 反馈控制最大次数