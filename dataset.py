import os
import pickle
import numpy as np
from config.config import Config
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from copy import deepcopy
import scipy.io as sio

class sleep_emotion():
    def __init__(self, subC:Config, glC:Config):
        self.eeg_path = glC.paths['eeg_feature_path']
        self.eye_path = glC.paths['eye_feature_path']
        self.subC = subC
        self.glC = glC
        stimuli = self.glC['stimuli']
        self.clipID_per_fold = stimuli['clipID_per_fold']
        self.clipID_label_appearence_order = stimuli['clipID_label_appearence_order']

    def get_data_sub_dependence(self, sessionID, foldID, subID, norm='standard'):
        eeg_path = self.eeg_path.format(sessionID+1)
        eye_path = self.eye_path.format(sessionID+1)

        subInfo = self.subC.info
        subName = subInfo[subID]['name']
        subDate = subInfo[subID]['date'][sessionID]
        fileName = subName + '_' + str(subDate) + '.npy'

        stimuli = self.glC['stimuli']
        clipIndex = stimuli['clipID_per_fold'][sessionID]
        labels = stimuli['label'][sessionID]
        foldNum = stimuli['n_fold']

        n_fold = list(range(foldNum))
        testIndex = clipIndex[foldID]
        n_fold.remove(foldID)
        trainIndex = []
        for i in n_fold:
            trainIndex.extend(clipIndex[i])
        print(trainIndex)
        print(testIndex)
        eeg_Data = np.load(os.path.join(eeg_path, fileName), allow_pickle=True).item()
        eye_Data = np.load(os.path.join(eye_path, fileName), allow_pickle=True).item()

        trainData_eeg = []
        testData_eeg = []
        trainData_eye = []
        testData_eye = []
        trainLabel = []
        testLabel = []
        trainClipLabel = []
        testClipLabel = []

        for i in range(len(trainIndex)):
            clipID = trainIndex[i]

            tempData_eeg = eeg_Data[f'clip_{clipID+1}']  #5,62
            trainData_eeg.append(tempData_eeg)
            tempData_eye = eye_Data[f'clip_{clipID+1}']
            trainData_eye.append(tempData_eye)
            assert tempData_eye.shape[0] == tempData_eeg.shape[0]

            tempLabel = labels[clipID]
            trainLabel.append(np.full(tempData_eeg.shape[0],tempLabel))

            trainClipLabel.append(np.full(tempData_eeg.shape[0],clipID+1))

        trainData_eye = np.vstack(trainData_eye)
        trainData_eeg = np.vstack(trainData_eeg)
        trainData_eeg = np.reshape(trainData_eeg, (-1, 310))
        trainLabel = np.hstack(trainLabel)
        trainClipLabel = np.hstack(trainClipLabel)


        for i in range(len(testIndex)):
            clipID = testIndex[i]

            tempData_eeg = eeg_Data[f'clip_{clipID + 1}']
            testData_eeg.append(tempData_eeg)
            tempData_eye = eye_Data[f'clip_{clipID + 1}']
            testData_eye.append(tempData_eye)
            assert tempData_eye.shape[0] == tempData_eeg.shape[0]

            tempLabel = labels[clipID]
            testLabel.append(np.full(tempData_eeg.shape[0], tempLabel))

            testClipLabel.append(np.full(tempData_eeg.shape[0],clipID+1))


        testData_eye = np.vstack(testData_eye)
        testData_eeg = np.vstack(testData_eeg)
        testData_eeg = np.reshape(testData_eeg, (-1, 310))
        testLabel = np.hstack(testLabel)
        testClipLabel = np.hstack(testClipLabel)



        norm_trainData_eye = deepcopy(trainData_eye)
        norm_trainData_eeg = deepcopy(trainData_eeg)

        trainData_eye = self._normalize(trainData_eye, norm_trainData_eye, norm)
        trainData_eeg = self._normalize(trainData_eeg, norm_trainData_eeg, norm)
        testData_eye = self._normalize(testData_eye, norm_trainData_eye, norm)
        testData_eeg = self._normalize(testData_eeg, norm_trainData_eeg, norm)

        return trainData_eeg, trainData_eye, testData_eeg, testData_eye, trainLabel, testLabel, trainClipLabel, testClipLabel 



    def get_data_cross_sub(self, sessionID, *subIDs, norm='standard'):
        eeg_path = self.eeg_path.format(sessionID+1)
        eye_path = self.eye_path.format(sessionID+1)

        subInfo = self.subC.info
        stimuli = self.glC['stimuli']
        labels = stimuli['label'][sessionID]
        n_clip = stimuli['n_clip']
        clipIDs = list(range(n_clip))

        test_subIDs = subIDs
        train_subIDs = list(range(len(subInfo)))
        for tempID in subIDs:
            train_subIDs.remove(tempID)

        trainData_eeg = []
        testData_eeg = []
        trainData_eye = []
        testData_eye = []
        trainLabel = []
        testLabel = []
        trainClipLabel = []
        testClipLabel = []
        trainSubLabel = []

        for test_subID in test_subIDs:
            subName = subInfo[test_subID]['name']
            subDate = subInfo[test_subID]['date'][sessionID]
            fileName = subName + '_' + str(subDate) + '.npy'

            eeg_Data = np.load(os.path.join(eeg_path, fileName), allow_pickle=True).item()
            temp_eeg_Data,concat_clip = self._concat_clip_data(eeg_Data, clipIDs)
            testData_eeg.append(temp_eeg_Data)
            testClipLabel.append(concat_clip)
            
            eye_Data = np.load(os.path.join(eye_path, fileName), allow_pickle=True).item()
            temp_eye_Data,_ = self._concat_clip_data(eye_Data, clipIDs)
            testData_eye.append(temp_eye_Data)

            assert temp_eeg_Data.shape[0] == temp_eye_Data.shape[0]
            testLabel.append(self._concat_clip_label(eye_Data, labels, clipIDs))

        for subLabel,train_subID in enumerate(train_subIDs):
            subName = subInfo[train_subID]['name']
            subDate = subInfo[train_subID]['date'][sessionID]
            fileName = subName + '_' + str(subDate) + '.npy'

            eeg_Data = np.load(os.path.join(eeg_path, fileName), allow_pickle=True).item()
            temp_eeg_Data,concat_clip = self._concat_clip_data(eeg_Data, clipIDs)
            trainData_eeg.append(temp_eeg_Data)
            trainClipLabel.append(concat_clip)
            trainSubLabel.append(np.full(temp_eeg_Data.shape[0],subLabel))
            eye_Data = np.load(os.path.join(eye_path, fileName), allow_pickle=True).item()
            temp_eye_Data,_ = self._concat_clip_data(eye_Data, clipIDs)
            trainData_eye.append(temp_eye_Data)

            assert temp_eeg_Data.shape[0] == temp_eye_Data.shape[0]
            trainLabel.append(self._concat_clip_label(eye_Data, labels, clipIDs))


        testData_eye = np.vstack(testData_eye)
        testData_eeg = np.vstack(testData_eeg)
        testData_eeg = np.reshape(testData_eeg, (-1, 310))
        testLabel = np.hstack(testLabel)
        testClipLabel = np.hstack(testClipLabel)
        trainData_eye = np.vstack(trainData_eye)
        trainData_eeg = np.vstack(trainData_eeg)
        trainData_eeg = np.reshape(trainData_eeg, (-1, 310))
        trainLabel = np.hstack(trainLabel)
        trainClipLabel = np.hstack(trainClipLabel)
        trainSubLabel = np.hstack(trainSubLabel)

        norm_trainData_eye = deepcopy(trainData_eye)
        norm_trainData_eeg = deepcopy(trainData_eeg)

        trainData_eye = self._normalize(trainData_eye, norm_trainData_eye, norm)
        trainData_eeg = self._normalize(trainData_eeg, norm_trainData_eeg, norm)
        testData_eye = self._normalize(testData_eye, norm_trainData_eye, norm)
        testData_eeg = self._normalize(testData_eeg, norm_trainData_eeg, norm)

        return trainData_eeg, trainData_eye, testData_eeg, testData_eye, trainLabel, testLabel, trainClipLabel, testClipLabel,trainSubLabel




    def _concat_clip_data(self, data, clipIDs):
        concat_data,concat_clip = [],[]
        for clipID in clipIDs:
            tempData = data[f'clip_{clipID + 1}']
            concat_data.append(tempData)
            concat_clip.append(np.full(tempData.shape[0],clipID+1))
        concat_data = np.vstack(concat_data)
        concat_clip = np.hstack(concat_clip)
        return concat_data,concat_clip

    def _concat_clip_label(self, data, labels, clipIDs):
        concat_label = []
        for clipID in clipIDs:
            sampleNum = data[f'clip_{clipID + 1}'].shape[0]
            tempLabel = labels[clipID]
            concat_label.append(np.full(sampleNum, tempLabel))
        concat_label = np.hstack(concat_label)
        return concat_label

    def _normalize(self, data, trainData, norm='standard'):
        if norm == 'standard':
            scaler = StandardScaler()
        elif norm == 'minmax':
            scaler = MinMaxScaler()
        scaler.fit(trainData)
        normData = scaler.transform(data)
        return normData


