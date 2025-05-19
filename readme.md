# ReLF
This is the code of ReLF:"Investigating the effects of sleep conditions on emotion responses with EEG signals and eye movements".

Please cite:
>@article{Li2025Investigating,\
title={Investigating the effects of sleep conditions on emotion responses with EEG signals and eye movements},\
  author={Li, Ziyi and Tao, Le-Yan and Ma, Rui-Xiao and Zheng, Wei-Long and Lu, Bao-Liang},\
  journal={IEEE Transactions on Affective Computing},\
  year={2025},\
  publisher={IEEE}
}

# Requirements
`pip install -r requirements.txt`

# Dataset
1. Apply SEED-SD through: https://bcmi.sjtu.edu.cn/home/seed/index.html
2. For data preprocessing and data splitting information, see the `config` folder. The code can be found in `dataset.py`.

# Pre-training and Fine-tuning
1. The code for pre-training model is in `ReLF_eeg_eye_pretrain.py`. The PyTorch Dataset implementation for subject-dependent and cross-subject pre-training can be found in `demo_pretrain_dataset.py`.
2. The code for multimodal fine-tuning model is in `ReLF_eeg_eye_finetune.py`. Please load the pre-trained weights before fine-tuning.
3. The codes for single-modal prompt tuning models are in `ReLF_eeg_finetune.py` and `ReLF_eye_finetune.py`. Same as multimodal fine-tuning, please load the pre-trained weights before fine-tuning.



