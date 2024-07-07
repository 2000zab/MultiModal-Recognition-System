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