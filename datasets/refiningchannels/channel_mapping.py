import mne
import numpy as np

def interpolate_to_58_channels(eeg_data, orig_ch_names, orig_ch_positions=None):
    """
    将任意通道数的EEG数据插值到58通道标准布局
    
    参数:
        eeg_data: 原始EEG数据，形状为[channels, samples] 或 [n_epochs, n_channels, n_samples]
        orig_ch_names: 原始数据的通道名称列表
        orig_ch_positions: 原始通道的3D位置（如果没有，将使用标准10-20位置）
    
    返回:
        插值后的58通道EEG数据
    """
    # EEGPT标准通道列表 - 使用正确的大小写
    target_ch_names = [
        'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 
        'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 
        'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 
        'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 
        'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 
        'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'
    ]
    
    # 创建标准10-20系统的montage
    standard_montage = mne.channels.make_standard_montage('standard_1020')
    
    # 处理数据维度
    if len(eeg_data.shape) == 3:  # [epochs, channels, samples]
        n_epochs, n_channels, n_samples = eeg_data.shape
        data_2d = eeg_data.reshape(n_epochs * n_samples, n_channels).T  # 转为[channels, samples]
        is_3d = True
    else:
        if eeg_data.shape[0] == len(orig_ch_names):  # [channels, samples]
            data_2d = eeg_data
        else:  # [samples, channels]
            data_2d = eeg_data.T
        is_3d = False
        n_channels, n_samples = data_2d.shape
        n_epochs = 1
    
    # 标准化原始通道名称 (确保第一个字母大写，其余小写)
    normalized_ch_names = []
    for ch in orig_ch_names:
        if ch.upper() in ('FP1', 'FP2', 'FPZ'):
            normalized_ch_names.append(ch.capitalize())
        elif ch.upper() in ('FCZ', 'CPZ', 'POZ'):
            normalized_ch_names.append(ch[:-1] + 'z')
        elif ch.upper() in ('FZ', 'CZ', 'PZ', 'OZ'):
            normalized_ch_names.append(ch[0] + 'z')
        else:
            normalized_ch_names.append(ch)
    
    # 创建原始Info对象
    orig_info = mne.create_info(normalized_ch_names, sfreq=250, ch_types='eeg')
    
    try:
        # 尝试设置蒙太奇，允许未找到的通道
        orig_info.set_montage(standard_montage, on_missing='warn')
    except Exception as e:
        print(f"警告：设置原始蒙太奇时出错: {e}")
        print("将尝试使用更简单的方法进行插值...")
    
    # 创建目标Info对象，确保所有通道都存在于标准montage中
    target_info = mne.create_info(target_ch_names, sfreq=250, ch_types='eeg')
    
    try:
        target_info.set_montage(standard_montage, on_missing='ignore')
    except Exception as e:
        print(f"错误：设置目标蒙太奇时出错: {e}")
        raise ValueError("无法设置目标蒙太奇，请检查通道名称是否正确")
    
    # 创建临时原始对象
    temp_raw = mne.io.RawArray(data_2d, orig_info)
    
    # 尝试使用不同的插值方法
    try:
        # 方法1：使用interpolate_bads
        interp_raw = temp_raw.copy()
        for ch_name in target_ch_names:
            if ch_name not in normalized_ch_names:
                interp_raw.info['bads'].append(ch_name)
        
        interp_raw = interp_raw.interpolate_bads(reset_bads=True, origin='auto', mode='accurate')
        interp_data = interp_raw.get_data()
        
        # 确保有58个通道
        if interp_data.shape[0] != len(target_ch_names):
            raise ValueError(f"插值后通道数不正确：{interp_data.shape[0]}，应为{len(target_ch_names)}")
            
    except Exception as e:
        print(f"方法1失败: {e}")
        
        try:
            # 方法2：使用球面样条插值
            from mne.channels import make_standard_montage
            from mne.channels.interpolation import _make_interpolation_matrix
            
            # 获取标准通道位置
            montage = make_standard_montage('standard_1020')
            ch_pos = montage.get_positions()['ch_pos']
            
            # 准备要使用的原始通道位置
            orig_pos = np.array([ch_pos[ch] for ch in normalized_ch_names if ch in ch_pos])
            orig_names = [ch for ch in normalized_ch_names if ch in ch_pos]
            
            if len(orig_names) < 16:
                raise ValueError("可用于插值的通道太少，至少需要16个")
            
            # 准备目标通道位置    
            target_pos = np.array([ch_pos[ch] for ch in target_ch_names if ch in ch_pos])
            
            # 计算插值矩阵
            interpolation = _make_interpolation_matrix(orig_pos, target_pos)
            
            # 只选择有位置的原始通道数据
            valid_ch_indices = [i for i, ch in enumerate(normalized_ch_names) if ch in orig_names]
            valid_data = data_2d[valid_ch_indices]
            
            # 应用插值
            interp_data = interpolation @ valid_data
            
        except Exception as e:
            print(f"方法2失败: {e}")
            
            # 最后的备选方法：简单地将缺失通道填零并匹配已知通道
            print("尝试最简单的方法：直接复制已知通道并填充缺失通道")
            interp_data = np.zeros((len(target_ch_names), n_samples))
            
            for i, target_ch in enumerate(target_ch_names):
                # 寻找最匹配的原始通道
                if target_ch in normalized_ch_names:
                    src_idx = normalized_ch_names.index(target_ch)
                    interp_data[i] = data_2d[src_idx]
                else:
                    # 找到最接近的通道
                    target_ch_lower = target_ch.lower()
                    for j, src_ch in enumerate(normalized_ch_names):
                        if src_ch.lower() == target_ch_lower:
                            interp_data[i] = data_2d[j]
                            break
    
    # 重塑回原始格式
    if is_3d:
        interp_data = interp_data.reshape(len(target_ch_names), n_epochs, n_samples)
        interp_data = interp_data.transpose(1, 0, 2)  # [epochs, channels, samples]
    
    return interp_data, target_ch_names


def preprocess_eeg_for_denoising(eeg_data, ch_names):
    """预处理EEG数据用于去噪"""
    # 检查数据维度，确保是[channels, samples]格式
    if len(eeg_data.shape) == 2 and eeg_data.shape[0] != len(ch_names):
        eeg_data = eeg_data.T  # 转置为[channels, samples]
    
    # 检查通道数是否需要插值
    if len(ch_names) != 58:
        print(f"原始通道数({len(ch_names)})与EEGPT模型要求(58)不匹配，执行空间插值...")
        interpolated_data, _ = interpolate_to_58_channels(eeg_data, ch_names)
        return interpolated_data
    else:
        return eeg_data


if __name__ == "__main__":
    # 加载BDF文件
    raw = mne.io.read_raw_bdf('./datasets/your_data/cai mei shuang open0506/data.bdf', preload=True)
    print(f"原始数据信息: {raw.info}")
    
    # 选择EEG通道
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    eeg_data = raw.get_data()[eeg_picks]
    
    print(f"提取的EEG通道数: {len(ch_names)}")
    print(f"通道名称: {ch_names}")
    
    # 使用适配函数处理数据
    adapted_eeg_data = preprocess_eeg_for_denoising(eeg_data, ch_names)
    
    # 验证通道数是否正确
    print(f"处理后通道数: {adapted_eeg_data.shape[0]}")  # 应该是58
    
    # 调整采样率
    sampling_rate = raw.info['sfreq']
    if sampling_rate != 256:
        print(f"重采样从 {sampling_rate}Hz 到 256Hz")
        from scipy import signal
        new_length = int(adapted_eeg_data.shape[1] * (256 / sampling_rate))
        adapted_eeg_data = signal.resample(adapted_eeg_data, new_length, axis=1)
    
    # 确保长度为256*4=1024个时间点
    target_length = 256 * 4
    if adapted_eeg_data.shape[1] != target_length:
        print(f"调整数据长度到 {target_length} 时间点")
        if adapted_eeg_data.shape[1] > target_length:
            adapted_eeg_data = adapted_eeg_data[:, :target_length]
        else:
            padding = np.zeros((adapted_eeg_data.shape[0], target_length - adapted_eeg_data.shape[1]))
            adapted_eeg_data = np.concatenate([adapted_eeg_data, padding], axis=1)
    
    print(f"最终数据形状: {adapted_eeg_data.shape}")  # 应该是[58, 1024]
    
    # 可选：将处理后的数据保存为numpy数组
    import os
    folder_name = os.path.basename(os.path.dirname('./datasets/your_data/cai mei shuang open0506/data.bdf'))
    save_path = f'./datasets/your_data/{folder_name}_processed.npy'
    np.save(save_path, adapted_eeg_data)
    print(f"处理完成，数据已保存至 {save_path}")