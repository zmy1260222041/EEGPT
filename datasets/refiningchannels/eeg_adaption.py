from .channel_mapping import interpolate_to_58_channels
import numpy as np
import mne

def preprocess_eeg_for_denoising(eeg_data, ch_names):
    """预处理EEG数据用于去噪"""
    # 检查通道数是否需要插值
    if len(ch_names) != 58:
        print(f"原始通道数({len(ch_names)})与EEGPT模型要求(58通道)不匹配，执行空间插值...")
        interpolated_data, _ = interpolate_to_58_channels(eeg_data, ch_names)
        return interpolated_data
    else:
        return eeg_data
    

# 1. 加载您的EEG数据
# 例如：使用MNE加载.bdf文件
raw = mne.io.read_raw_bdf('yourdata/cai mei shuang open0506/data.bdf', preload=True)

# 2. 获取数据和通道名称
eeg_data = raw.get_data()  # 获取形状为[channels, samples]的数据
ch_names = raw.ch_names    # 获取通道名称列表

# 3. 使用适配函数处理数据
adapted_eeg_data = preprocess_eeg_for_denoising(eeg_data, ch_names)

# 4. 验证通道数是否正确
print(f"原始通道数: {len(ch_names)}")
print(f"处理后通道数: {adapted_eeg_data.shape[0]}")  # 应该是58

# 5. 进一步处理 - 如果需要调整采样率或时间长度
# 假设EEGPT要求256Hz采样率和4秒数据长度
sampling_rate = raw.info['sfreq']
if sampling_rate != 256:
    # 重采样到256Hz
    from scipy import signal
    new_length = int(eeg_data.shape[1] * (256 / sampling_rate))
    adapted_eeg_data = signal.resample(adapted_eeg_data, new_length, axis=1)

# 确保长度为256*4=1024个时间点
if adapted_eeg_data.shape[1] != 1024:
    # 通过截断或填充调整长度
    if adapted_eeg_data.shape[1] > 1024:
        adapted_eeg_data = adapted_eeg_data[:, :1024]
    else:
        padding = np.zeros((adapted_eeg_data.shape[0], 1024 - adapted_eeg_data.shape[1]))
        adapted_eeg_data = np.concatenate([adapted_eeg_data, padding], axis=1)

# 现在adapted_eeg_data的形状应该是[58, 1024]，可以输入到EEGPT模型中
print(f"最终数据形状: {adapted_eeg_data.shape}")  # 应该是[58, 1024]