from config.config import Config, get_param_sets
from dataset import sleep_emotion
from ReLF_eeg_eye_pretrain import ReLF

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

class SeqDataset(Dataset):
    def __init__(self, eeg,eye,clip,time_window):
        super().__init__()
        self.eeg=eeg
        self.eye=eye
        self.clip=clip
        self.time_window=time_window

        self.dic = {}
        self.cnt = 0
        for i in range(clip.shape[0] - time_window):
            if clip[i] == clip[i + time_window]:
                self.dic[self.cnt] = i
                self.cnt += 1        
 
    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        start = self.dic[index]
        end = start + self.time_window
        return self.eeg[start:end], self.eye[start:end]
    


if __name__ == '__main__':
    ###########  subject-dependence Dataset #############
    sleep_sessionID = 0 # sessionID: 0 SD, 1 SR, 2 NS
    foldID = 0 # foldID: 0, 1, 2
    time_window = 5 

    save_path = 'your_path'
    output_dir=Path(save_path)/  ('fold_%s' % foldID)

    subj_config_path = Path('./config/subjects.yaml')
    glob_config_path = Path('./config/global.yaml')
    subC = Config(subj_config_path)
    glC = Config(glob_config_path)
    Data = sleep_emotion(subC, glC)

    print(f'========sleep_sessionID {sleep_sessionID}==============')
    print(f'========foldID {foldID}==============')
    eeg_input_dim = 310
    eye_input_dim = 50
    eeg=[]
    eye=[]
    clip=[]
    for subID in range(40):
        train_eeg, train_eye, _, _,_, _, trainClipLabel,_ = Data.get_data_sub_dependence(sessionID=sleep_sessionID ,foldID=foldID, subID=subID, norm='standard')
        eeg.append(train_eeg)
        eye.append(train_eye)
        clip.append(trainClipLabel)
    eeg=np.vstack(eeg)
    eye=np.vstack(eye)
    clip=np.hstack(clip)
    print(eeg.shape)
    print(eye.shape)

    datasets=SeqDataset(eeg=eeg,eye=eye,clip=clip,time_window=time_window)


    ###########  cross-subject Dataset #############
    sleep_sessionID = 0 # sessionID: 0 SD, 1 SR, 2 NS
    subID = 0 # range(40)
    time_window = 5 

    save_path = 'your_path'
    output_dir=Path(save_path)/('sub_%s' % subID)

    subj_config_path = Path('./config/subjects.yaml')
    glob_config_path = Path('./config/global.yaml')
    subC = Config(subj_config_path)
    glC = Config(glob_config_path)
    Data = sleep_emotion(subC, glC)

    print(f'========sleep_sessionID {sleep_sessionID}==============')
    print(f'========subID {subID}==============')
    eeg_input_dim = 310
    eye_input_dim = 50
    eeg=[]
    eye=[]
    clip=[]
    train_eeg, train_eye, _, _,_, _, trainClipLabel,_,_ = Data.get_data_cross_sub(sleep_sessionID, subID)
    eeg.append(train_eeg)
    eye.append(train_eye)
    clip.append(trainClipLabel)
    eeg=np.vstack(eeg)
    eye=np.vstack(eye)
    clip=np.hstack(clip)
    print(eeg.shape)
    print(eye.shape)

    datasets=SeqDataset(eeg=eeg,eye=eye,clip=clip,time_window=time_window)


