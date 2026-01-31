import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pyedflib
import re
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import tqdm
from scipy.signal import stft, hamming, resample
import matplotlib.pyplot as plt
# 新增：导入 Tuple 用于类型注解（适配 Python 3.8 及以下）
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 全局配置（仅保留目标信号）
CONFIG = {
    'target_fs': 100,  # 目标采样率（Hz）
    'segment_seconds': 30,  # 片段时长（固定30秒，与标签对齐）
    'target_signal_length': 3000,  # 目标信号长度（100Hz×30s）
    'context_len': 5,  # 上下文窗口大小（目标epoch±2）
    'modal_missing_rate': 0.2,  # 模态缺失率
    'stft_nfft': 256,  # STFT点数
    'stft_win_seconds': 2,  # STFT窗口时长（秒）
    'stft_overlap_ratio': 0.5,  # STFT重叠率
    'essential_signals': {  # 必需信号（仅保留EEG C3-A2和EOG LOC-A2）
        'EEG C3-A2', 'EOG LOC-A2'
    },
    'target_modals': ['EEG', 'EOG'],  # 模型输入模态
    'sleep_stage_mapping': {  # AASM标准映射
        'NotScored': 0, 'Wake': 1, 'NonREM1': 2, 'NonREM2': 3, 'NonREM3': 4, 'REM': 5
    },
    'model_stage_mapping': {  # 模型输入标签映射（0-4）
        1: 0,  # Wake → 0
        2: 1,  # NonREM1 → 1
        3: 2,  # NonREM2 → 2
        4: 3,  # NonREM3 → 3
        5: 4   # REM → 4
    }
}

class BioSignalProcessor:
    """信号预处理核心类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_fs = config['target_fs']
        self.target_len = config['target_signal_length']
        # STFT参数
        self.stft_win_points = int(config['target_fs'] * config['stft_win_seconds'])
        self.stft_overlap_points = int(self.stft_win_points * config['stft_overlap_ratio'])
        self.stft_nfft = config['stft_nfft']
        self.hamming_win = hamming(self.stft_win_points)

    def resample_to_target_fs(self, signal: np.ndarray, original_fs: float) -> np.ndarray:
        """重采样到目标采样率"""
        if np.isclose(original_fs, self.target_fs, atol=1e-2):
            return signal
        target_length = int(round(len(signal) * (self.target_fs / original_fs)))
        if target_length <= 0:
            return np.zeros(self.target_len)
        return resample(signal, target_length)

    def remove_outliers(self, signal: np.ndarray) -> np.ndarray:
        """IQR异常值去除"""
        q1 = np.percentile(signal, 25)
        q3 = np.percentile(signal, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return np.clip(signal, lower_bound, upper_bound)

    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Z-score 标准化：零均值、单位方差（与论文一致）"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-8:
            # 近似平坦信号在后续会被质量控制剔除，这里返回零向量以避免数值问题
            return np.zeros_like(signal)
        return (signal - mean) / (std + 1e-8)

    
    def is_low_quality(self, signal: np.ndarray) -> bool:
        """低质量波形检测（导联脱落/近似平坦/明显基线漂移等）
        说明：该检测用于实现论文中“剔除低质量波形”的预处理步骤。
        """
        if signal is None or len(signal) < 10:
            return True

        sig = np.asarray(signal, dtype=np.float32)
        std = np.std(sig)
        if std < 1e-6:
            # 近似平坦（可能导联脱落/断开）
            return True

        # 近似常数导数比例（长时间不变化）
        diff = np.abs(np.diff(sig))
        if np.mean(diff < 1e-6) > 0.95:
            return True

        # 基线漂移/阶跃：首尾均值差过大（相对标准差）
        n = len(sig)
        head = np.mean(sig[: max(1, n // 10)])
        tail = np.mean(sig[-max(1, n // 10):])
        if np.abs(tail - head) > 5.0 * std:
            return True

        # 异常饱和：极端值占比过高（传感器饱和/剪切）
        p01, p99 = np.percentile(sig, [1, 99])
        if (p99 - p01) < 1e-6:
            return True
        frac_extreme = np.mean((sig <= p01) | (sig >= p99))
        if frac_extreme > 0.20:
            return True

        return False

    def process_single_signal(self, signal: np.ndarray, original_fs: float) -> np.ndarray:
        """完整信号处理流程：重采样→低质量波形剔除→Z-score标准化→长度对齐"""
        # 1) 重采样
        resampled = self.resample_to_target_fs(signal, original_fs)

        # 2) 低质量波形检测（导联脱落/平坦/明显漂移等）——与论文描述一致
        if self.is_low_quality(resampled):
            raise ValueError("Low-quality waveform detected (e.g., lead disconnection/baseline drift/flatline).")

        # 3) Z-score 标准化（零均值、单位方差）
        normalized = self.normalize_signal(resampled)
        # 长度对齐
        if len(normalized) > self.target_len:
            return normalized[:self.target_len]
        elif len(normalized) < self.target_len:
            pad = self.target_len - len(normalized)
            return np.pad(normalized, (0, pad), mode='constant')
        return normalized

    def generate_time_freq_image(self, signal: np.ndarray) -> np.ndarray:
        """生成时频图像（STFT）"""
        # STFT
        f, t, Zxx = stft(
            signal, fs=self.target_fs, window=self.hamming_win,
            nperseg=self.stft_win_points, noverlap=self.stft_overlap_points,
            nfft=self.stft_nfft, return_onesided=True
        )
        # 对数幅度谱
        amp_spec = 20 * np.log10(np.abs(Zxx) + 1e-8)
        # 归一化到[0,1]
        amp_spec = (amp_spec - amp_spec.min()) / (amp_spec.max() - amp_spec.min() + 1e-8)
        return amp_spec  # (129, 29)

class XMLSleepStageParser:
    """XML睡眠阶段标签解析类（修改：仅从UserStaging提取）"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stage_mapping = config['sleep_stage_mapping']

    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """提取XML命名空间"""
        namespaces = {}
        if root.tag.startswith("{"):
            ns_uri = root.tag.split("}")[0][1:]
            namespaces["ns"] = ns_uri
        return namespaces

    def _print_xml_structure(self, node: ET.Element, depth: int = 0, max_depth: int = 3):
        """打印XML结构（调试用）"""
        if depth > max_depth:
            return
        indent = "  " * depth
        print(f"{indent}- {node.tag}")
        for i, child in enumerate(node[:5]):
            self._print_xml_structure(child, depth + 1, max_depth)
        if len(node) > 5:
            print(f"{indent}  ... 还有 {len(node) - 5} 个子节点省略 ...")

    # 修复：tuple[list, float, float] → Tuple[List[Dict[str, Any]], float, float]
    def parse(self, xml_path: str) -> Tuple[List[Dict[str, Any]], float, float]:
        """解析XML，仅从UserStaging提取标签，无则抛出异常"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            raise ValueError(f"解析XML失败: {str(e)}")
        
        namespaces = self._extract_namespaces(root)
        
        # 尝试多种可能的StagingData路径
        possible_paths = [
            ".//ns:StagingData", ".//ns:ScoringData/ns:StagingData",
            ".//ns:SleepStagingData", ".//ns:Staging", 
            ".//ns:ExpertStaging/ns:StageData", ".//ns:SleepScore/ns:StagingSegment",
            ".//StagingData", ".//ScoringData/StagingData",
            ".//SleepStagingData", ".//Staging",
            ".//ExpertStaging/StageData", ".//SleepScore/StagingSegment"
        ]
        
        staging_data = None
        for path in possible_paths:
            staging_data = root.find(path, namespaces)
            if staging_data is not None:
                print(f"找到StagingData节点，路径: {path}")
                break
        
        if staging_data is None:
            print("XML结构调试:")
            self._print_xml_structure(root)
            raise ValueError("未找到睡眠阶段数据节点")

        # 提取阶段数据（仅从UserStaging提取，无则抛出异常）
        stage_list = []
        first_wake_start = None
        
        def extract_stages(container: ET.Element) -> List[Tuple[str, float]]:
            stages = []
            neuro_stagings = container.findall(".//ns:NeuroAdultAASMStaging", namespaces)
            if not neuro_stagings:
                neuro_stagings = container.findall(".//NeuroAdultAASMStaging")
            
            for ns in neuro_stagings:
                stage_elems = ns.findall(".//ns:Stage", namespaces) or ns.findall(".//Stage")
                for elem in stage_elems:
                    stage_type = elem.get("Type")
                    if not stage_type or stage_type not in self.stage_mapping:
                        continue
                    try:
                        start = float(elem.get("Start", 0.0))
                        stages.append((stage_type, start))
                    except:
                        continue
            return stages
        
        # 仅从UserStaging提取，不尝试MachineStaging
        user_staging_node = staging_data.find(".//ns:UserStaging", namespaces) or staging_data.find(".//UserStaging")
        if not user_staging_node:
            raise ValueError("未找到UserStaging节点，跳过该受试者")
        
        user_stages = extract_stages(user_staging_node)
        if not user_stages:
            raise ValueError("UserStaging中无有效阶段数据，跳过该受试者")

        # 处理阶段数据
        for stage_type, start_time in user_stages:
            if stage_type == 'NotScored':
                continue
            num_label = self.stage_mapping[stage_type]
            if first_wake_start is None and stage_type == 'Wake':
                first_wake_start = start_time
            stage_list.append({
                'type': stage_type,
                'start': start_time,
                'num_label': num_label
            })
        
        # 处理无Wake阶段的情况
        if first_wake_start is None:
            first_wake_start = stage_list[0]['start']
            print(f"未找到Wake阶段，使用第一个阶段起始时间: {first_wake_start}s")
        
        # 排序并补充结束时间
        sorted_stages = sorted(stage_list, key=lambda x: x['start'])
        for i in range(len(sorted_stages)):
            if i < len(sorted_stages) - 1:
                sorted_stages[i]['end'] = sorted_stages[i+1]['start']
            else:
                sorted_stages[i]['end'] = sorted_stages[i]['start'] + 30  # 最后一个阶段默认30秒
        
        # 对齐到30秒整数倍
        def align_to_30s(time: float) -> float:
            return ((int(time) + 29) // 30) * 30
        
        valid_stages = []
        for stage in sorted_stages:
            stage['start'] = align_to_30s(stage['start'])
            stage['end'] = align_to_30s(stage['end'])
            if stage['end'] > stage['start']:
                valid_stages.append(stage)
        
        max_stage_end = max(stage['end'] for stage in valid_stages)
        first_wake_start = align_to_30s(first_wake_start)

        # 打印阶段信息
        print(f"\n解析到 {len(valid_stages)} 个有效阶段（仅来自UserStaging）:")
        stage_count = defaultdict(int)
        for stage in valid_stages:
            stage_count[stage['type']] += 1
            duration = stage['end'] - stage['start']
            print(f"  {stage['type']:8s}: {stage['start']:6.0f}s ~ {stage['end']:6.0f}s ({duration/60:.1f}min)")
        
        for stage, count in stage_count.items():
            print(f"  {stage}: {count}个片段")
        
        return valid_stages, first_wake_start, max_stage_end

class EDFSignalLoader:
    """EDF信号加载类（修改：仅加载EEG C3-A2和EOG LOC-A2）"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.essential_signals = config['essential_signals']  # 仅包含EEG C3-A2和EOG LOC-A2

    def _load_edf_file(self, edf_path: str) -> Dict[str, Dict[str, Any]]:
        """加载单个EDF文件（仅保留必需信号）"""
        signal_data = defaultdict(dict)
        try:
            with pyedflib.EdfReader(edf_path) as f:
                signal_labels = [label.strip() for label in f.getSignalLabels()]
                for i in range(len(signal_labels)):
                    label = signal_labels[i]
                    # 只处理必需信号（EEG C3-A2和EOG LOC-A2）
                    if label not in self.essential_signals:
                        continue
                    signal_data[label]['data'] = f.readSignal(i)
                    signal_data[label]['fs'] = f.getSampleFrequency(i)
                    signal_data[label]['label'] = label
        except Exception as e:
            print(f"加载EDF失败 {edf_path}: {str(e)}")
        return signal_data

    def _merge_signals(self, signal_accumulator: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """合并多个EDF文件的信号（仅合并必需信号）"""
        merged_signals = {}
        for label, info in signal_accumulator.items():
            if len(info['data_list']) == 0:
                continue
            # 拼接信号数据
            merged_data = np.concatenate(info['data_list'])
            merged_signals[label] = {
                'data': merged_data,
                'fs': info['fs'],
                'total_duration': len(merged_data) / info['fs']
            }
        return merged_signals

    def load(self, edf_dir: str) -> Dict[str, Dict[str, Any]]:
        """加载目录下所有EDF文件并合并（仅保留必需信号）"""
        edf_files = sorted([f for f in os.listdir(edf_dir) if f.endswith(".edf")])
        if not edf_files:
            raise ValueError(f"EDF目录 {edf_dir} 无有效文件")
        
        # 信号累积器：处理多文件拼接
        signal_accumulator = defaultdict(lambda: {'data_list': [], 'fs': None})
        
        print(f"加载 {len(edf_files)} 个EDF文件（仅保留EEG C3-A2和EOG LOC-A2）...")
        for edf_file in edf_files:
            edf_path = os.path.join(edf_dir, edf_file)
            file_signals = self._load_edf_file(edf_path)
            
            for label, sig_info in file_signals.items():
                if signal_accumulator[label]['fs'] is None:
                    # 首次出现，记录采样率
                    signal_accumulator[label]['fs'] = sig_info['fs']
                    signal_accumulator[label]['data_list'].append(sig_info['data'])
                else:
                    # 校验采样率一致性（必需信号严格校验）
                    if not np.isclose(signal_accumulator[label]['fs'], sig_info['fs'], atol=1e-2):
                        raise ValueError(
                            f"信号 {label} 采样率不一致: "
                            f"前文件 {signal_accumulator[label]['fs']}Hz, "
                            f"当前文件 {sig_info['fs']}Hz"
                        )
                    signal_accumulator[label]['data_list'].append(sig_info['data'])
        
        # 合并信号
        merged_signals = self._merge_signals(signal_accumulator)
        
        # 校验必需信号是否齐全（必须同时有EEG C3-A2和EOG LOC-A2）
        missing_signals = self.essential_signals - set(merged_signals.keys())
        if missing_signals:
            raise ValueError(f"缺失必需信号: {missing_signals}（仅支持EEG C3-A2和EOG LOC-A2）")
        
        # 打印加载信息
        print(f"\n成功加载 {len(merged_signals)} 个信号:")
        for label, sig in merged_signals.items():
            print(f"  {label:18s}: {sig['fs']:.1f}Hz, 时长: {sig['total_duration']/3600:.2f}h")
        
        return merged_signals

class SleepStudyProcessor:
    """完整睡眠研究处理器（修改：固定使用EEG C3-A2和EOG LOC-A2）"""
    def __init__(self, edf_dir: str, xml_path: str, config: Dict[str, Any]):
        self.edf_dir = edf_dir
        self.xml_path = xml_path
        self.config = config
        self.signal_loader = EDFSignalLoader(config)
        self.stage_parser = XMLSleepStageParser(config)
        self.signal_processor = BioSignalProcessor(config)
        
        # 加载数据
        self.merged_signals = self.signal_loader.load(edf_dir)  # 仅包含EEG C3-A2和EOG LOC-A2
        self.sleep_stages, self.first_wake_start, self.max_stage_end = self.stage_parser.parse(xml_path)  # 仅来自UserStaging
        
        # 提取目标模态（固定为EEG C3-A2和EOG LOC-A2）
        self.target_signals = self._select_target_modals()

    def _select_target_modals(self) -> Dict[str, Dict[str, Any]]:
        """选择目标模态信号（固定为EEG C3-A2和EOG LOC-A2）"""
        target_signals = {}
        
        # 强制选择EEG C3-A2（已在EDFSignalLoader中校验过存在）
        target_signals['EEG'] = self.merged_signals['EEG C3-A2']
        print(f"选择EEG信号: EEG C3-A2（固定）")
        
        # 强制选择EOG LOC-A2（已在EDFSignalLoader中校验过存在）
        target_signals['EOG'] = self.merged_signals['EOG LOC-A2']
        print(f"选择EOG信号: EOG LOC-A2（固定）")
        
        return target_signals

    def _get_segment_samples(self, modal: str, start_time: float, end_time: float) -> np.ndarray:
        """提取单个片段的信号数据"""
        sig_info = self.target_signals[modal]
        fs = sig_info['fs']
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        # 边界检查
        if start_sample < 0:
            start_sample = 0
        if end_sample > len(sig_info['data']):
            end_sample = len(sig_info['data'])
        
        if start_sample >= end_sample:
            return np.zeros(self.config['target_signal_length'])
        
        # 提取原始片段
        raw_segment = sig_info['data'][start_sample:end_sample]
        # 预处理
        processed = self.signal_processor.process_single_signal(raw_segment, fs)
        return processed

    def _match_segment_stage(self, start_time: float, end_time: float) -> int:
        """匹配片段的睡眠阶段标签"""
        for stage in self.sleep_stages:
            if stage['start'] <= start_time and end_time <= stage['end']:
                # 映射为模型标签
                return self.config['model_stage_mapping'].get(stage['num_label'], 0)
        return 0  # 未匹配到标签

    def generate_segments(self, overlap_seconds: int = 0, skip_first_n: int = 0) -> List[Dict[str, Any]]:
        """生成带标签的30秒片段"""
        segment_seconds = self.config['segment_seconds']
        if overlap_seconds < 0 or overlap_seconds >= segment_seconds:
            raise ValueError(f"重叠时长必须为0~{segment_seconds-1}秒")
        
        # 计算片段步长（秒）
        step_seconds = segment_seconds - overlap_seconds
        # 计算总片段数
        max_start_time = self.max_stage_end - segment_seconds
        if max_start_time < self.first_wake_start:
            raise ValueError("信号长度不足，无法生成片段")
        
        total_possible_segments = int((max_start_time - self.first_wake_start) / step_seconds) + 1
        if total_possible_segments <= skip_first_n:
            raise ValueError(f"跳过前{skip_first_n}个片段后无有效片段")
        
        # 生成片段
        segments = []
        print(f"\n生成片段：时长{segment_seconds}s，重叠{overlap_seconds}s，跳过前{skip_first_n}个")
        for seg_idx in range(total_possible_segments - skip_first_n):
            actual_idx = seg_idx + skip_first_n
            start_time = self.first_wake_start + actual_idx * step_seconds
            end_time = start_time + segment_seconds
            
            if end_time > self.max_stage_end:
                break
            
            # 提取EEG C3-A2和EOG LOC-A2片段（若检测到低质量波形则跳过该epoch）
            try:
                eeg_segment = self._get_segment_samples('EEG', start_time, end_time)
                eog_segment = self._get_segment_samples('EOG', start_time, end_time)
            except ValueError:
                # 论文中描述的“剔除低质量波形”
                continue
            
            # 匹配标签（仅来自UserStaging）
            stage_label = self._match_segment_stage(start_time, end_time)
            
            # 生成时频图像
            eeg_freq = self.signal_processor.generate_time_freq_image(eeg_segment)
            eog_freq = self.signal_processor.generate_time_freq_image(eog_segment)
            
            segments.append({
                'segment_id': seg_idx,
                'start_time': start_time,
                'end_time': end_time,
                'eeg_time': eeg_segment,  # (3000,) → EEG C3-A2
                'eog_time': eog_segment,  # (3000,) → EOG LOC-A2
                'eeg_freq': eeg_freq,     # (129, 29) → EEG C3-A2
                'eog_freq': eog_freq,     # (129, 29) → EOG LOC-A2
                'label': stage_label      # 0-4
            })
        
        print(f"成功生成 {len(segments)} 个有效片段")
        # 统计标签分布
        label_count = defaultdict(int)
        for seg in segments:
            label_count[seg['label']] += 1
        for label, count in sorted(label_count.items()):
            stage_name = [k for k, v in self.config['model_stage_mapping'].items() if v == label][0]
            print(f"  阶段 {stage_name} (标签{label}): {count}个片段")
        
        return segments

class SleepStagingDataset(Dataset):
    """PyTorch数据集接口"""
    def __init__(self, study_segments: List[Dict[str, Any]], config: Dict[str, Any]):
        self.segments = study_segments
        self.config = config
        self.context_len = config['context_len']
        self.half_context = self.context_len // 2
        self.modal_missing_rate = config['modal_missing_rate']
        self.num_segments = len(study_segments)

    # 修复：tuple[np.ndarray, np.ndarray, int] → Tuple[np.ndarray, np.ndarray, int]
    def _generate_context(self, seg_idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """生成上下文序列"""
        # 计算上下文索引范围
        start_idx = max(0, seg_idx - self.half_context)
        end_idx = min(self.num_segments, seg_idx + self.half_context + 1)
        
        # 填充边缘
        pad_left = self.half_context - seg_idx if seg_idx < self.half_context else 0
        pad_right = (seg_idx + self.half_context + 1) - self.num_segments if seg_idx + self.half_context >= self.num_segments else 0
        
        # 提取上下文片段
        context_segs = self.segments[start_idx:end_idx]
        
        # 左侧填充
        if pad_left > 0:
            context_segs = [context_segs[0]] * pad_left + context_segs
        # 右侧填充
        if pad_right > 0:
            context_segs = context_segs + [context_segs[-1]] * pad_right
        
        # 构建时域和频域序列（仅EEG C3-A2和EOG LOC-A2）
        time_seq = []
        freq_seq = []
        for seg in context_segs:
            # 时域：(2, 3000) → EEG C3-A2 + EOG LOC-A2
            time_data = np.stack([seg['eeg_time'], seg['eog_time']], axis=0)
            time_seq.append(time_data)
            # 频域：(2, 129, 29) → EEG C3-A2 + EOG LOC-A2
            freq_data = np.stack([seg['eeg_freq'], seg['eog_freq']], axis=0)
            freq_seq.append(freq_data)
        
        # 目标标签（中间片段的标签）
        target_label = context_segs[self.half_context]['label']
        
        return np.array(time_seq), np.array(freq_seq), target_label

    def _generate_missing_mask(self) -> np.ndarray:
        """生成模态缺失掩码"""
        mask = np.ones(2, dtype=np.int64)  # 1=存在，0=缺失
        if np.random.random() < self.modal_missing_rate:
            mask[0] = 0  # EEG C3-A2缺失
        if np.random.random() < self.modal_missing_rate:
            mask[1] = 0  # EOG LOC-A2缺失
        return mask

    def __len__(self) -> int:
        return self.num_segments

    # 修复：tuple[torch.Tensor, ...] → Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
            time_seq: 上下文时域信号 (5, 2, 3000) → (批次, 上下文, 模态(EEG/EOG), 信号长度)
            freq_seq: 上下文时频图像 (5, 2, 129, 29) → (批次, 上下文, 模态, 频率点, 时间窗)
            label: 目标标签 (1,)
            modal_mask: 模态缺失掩码 (2,) → (EEG, EOG)
            seg_idx: 片段索引 (1,)
        """
        # 生成上下文序列
        time_seq, freq_seq, label = self._generate_context(idx)
        
        # 生成缺失掩码
        modal_mask = self._generate_missing_mask()
        
        # 应用缺失掩码（置零）
        if modal_mask[0] == 0:
            time_seq[:, 0, :] = 0  # EEG C3-A2置零
            freq_seq[:, 0, :, :] = 0
        if modal_mask[1] == 0:
            time_seq[:, 1, :] = 0  # EOG LOC-A2置零
            freq_seq[:, 1, :, :] = 0
        
        # 转换为Tensor
        time_seq = torch.from_numpy(time_seq).float()
        freq_seq = torch.from_numpy(freq_seq).float()
        label = torch.tensor(label, dtype=torch.long)
        modal_mask = torch.tensor(modal_mask, dtype=torch.long)
        seg_idx = torch.tensor(idx, dtype=torch.long)
        
        return time_seq, freq_seq, label, modal_mask, seg_idx

# 批量处理多个受试者（支持按受试者划分训练集/测试集）
class MultiStudyDatasetGenerator:
    """批量生成多受试者数据集（严格过滤无UserStaging或信号不全的受试者）"""
    def __init__(self, root_dir: str, config: Dict[str, Any]):
        self.root_dir = root_dir
        self.config = config
        self.study_folders = self._find_valid_study_folders()

    def _find_valid_study_folders(self) -> List[str]:
        """查找包含EDF和XML的受试者文件夹（后续还要过滤信号和标签）"""
        valid_folders = []
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if not os.path.isdir(item_path):
                continue
            has_edf = any(f.endswith('.edf') for f in os.listdir(item_path))
            has_xml = any(f.endswith('.xml') for f in os.listdir(item_path))
            if has_edf and has_xml:
                valid_folders.append(item_path)
        
        if not valid_folders:
            raise ValueError(f"未找到有效受试者文件夹（需同时包含EDF和XML）")
        
        print(f"找到 {len(valid_folders)} 个包含EDF和XML的文件夹:")
        for folder in valid_folders:
            print(f"  - {os.path.basename(folder)}")
        
        return valid_folders

    def generate_single_study(self, folder_path: str) -> List[Dict[str, Any]]:
        """处理单个受试者（无UserStaging或信号不全则跳过）"""
        study_name = os.path.basename(folder_path)
        print(f"\n{'='*60}")
        print(f"处理受试者: {study_name}")
        print(f"{'='*60}")
        
        # 找到XML文件
        xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]
        if not xml_files:
            print(f"无XML文件，跳过该受试者")
            return []
        xml_path = os.path.join(folder_path, xml_files[0])
        
        try:
            # 处理该受试者（信号仅EEG C3-A2+EOG LOC-A2，标签仅UserStaging）
            study_processor = SleepStudyProcessor(
                edf_dir=folder_path,
                xml_path=xml_path,
                config=self.config
            )
            # 生成片段（无重叠，不跳过）
            segments = study_processor.generate_segments(
                overlap_seconds=0,
                skip_first_n=0
            )
            print(f"✅ 受试者 {study_name} 处理成功")
            return segments
        except Exception as e:
            print(f"❌ 受试者 {study_name} 处理失败（跳过）: {str(e)}")
            return []

    def generate_dataset(self, save_npy: bool = False, save_path: str = "./dataset") -> Dataset:
        """生成完整数据集（不划分）"""
        all_segments = []
        print(f"\n开始批量处理 {len(self.study_folders)} 个受试者...")
        print(f"过滤条件：1. 必须包含EEG C3-A2和EOG LOC-A2信号；2. 必须有UserStaging标签")
        
        for folder in tqdm.tqdm(self.study_folders, desc="处理进度"):
            segments = self.generate_single_study(folder)
            if segments:
                all_segments.extend(segments)
        
        if not all_segments:
            raise ValueError("所有受试者均不符合条件（无UserStaging标签或信号不全）")
        
        print(f"\n{'='*60}")
        print(f"批量处理完成")
        print(f"{'='*60}")
        print(f"符合条件的受试者生成总片段数: {len(all_segments)}")
        # 统计整体标签分布
        label_count = defaultdict(int)
        for seg in all_segments:
            label_count[seg['label']] += 1
        print("整体标签分布:")
        for label, count in sorted(label_count.items()):
            stage_name = [k for k, v in self.config['model_stage_mapping'].items() if v == label][0]
            print(f"  阶段 {stage_name} (标签{label}): {count}个片段 ({count/len(all_segments)*100:.1f}%)")
        
        # 保存为npy（可选）
        if save_npy:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "all_segments.npy"), all_segments)
            print(f"片段数据已保存至: {os.path.join(save_path, 'all_segments.npy')}")
        
        # 返回PyTorch数据集
        return SleepStagingDataset(all_segments, self.config)

    # 修复：tuple[Dataset, Dataset] → Tuple[Dataset, Dataset]
    def generate_split_dataset(
        self, 
        train_ratio: float = 0.8,  # 训练集比例
        random_seed: int = 2025,  # 随机种子（保证划分可复现）
        save_npy: bool = False, 
        save_path: str = "./dataset"
    ) -> Tuple[Dataset, Dataset]:
        """
        自动划分训练集/测试集（按受试者划分，避免数据泄露）
        仅包含：有UserStaging标签 + 同时有EEG C3-A2和EOG LOC-A2信号的受试者
        """
        # 1. 加载所有符合条件的受试者片段（记录每个受试者的片段映射）
        study_segment_map = {}  # key: 受试者名称, value: 该受试者的所有片段
        print(f"\n{'='*60}")
        print(f"开始加载所有符合条件的受试者数据...")
        print(f"过滤条件：1. 必须包含EEG C3-A2和EOG LOC-A2信号；2. 必须有UserStaging标签")
        print(f"{'='*60}")
        
        for folder in tqdm.tqdm(self.study_folders, desc="加载进度"):
            segments = self.generate_single_study(folder)
            if segments:
                study_name = os.path.basename(folder)
                study_segment_map[study_name] = segments
        
        if not study_segment_map:
            raise ValueError("无符合条件的受试者（无UserStaging标签或信号不全）")
        
        # 2. 按受试者划分训练集/测试集（关键：避免同一受试者跨集）
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        study_names = list(study_segment_map.keys())
        np.random.shuffle(study_names)  # 打乱受试者顺序
        
        # 划分边界（按比例计算训练集受试者数量）
        train_study_num = int(len(study_names) * train_ratio)
        train_studies = study_names[:train_study_num]
        test_studies = study_names[train_study_num:]
        
        # 收集训练集/测试集片段
        train_segments = []
        for study in train_studies:
            train_segments.extend(study_segment_map[study])
        
        test_segments = []
        for study in test_studies:
            test_segments.extend(study_segment_map[study])
        
        # 3. 打印划分详情
        print(f"\n{'='*60}")
        print(f"数据集划分结果（按受试者划分）")
        print(f"{'='*60}")
        print(f"符合条件的受试者总数：{len(study_names)}")
        print(f"训练集：{len(train_studies)} 个受试者 → {len(train_segments)} 个片段")
        print(f"测试集：{len(test_studies)} 个受试者 → {len(test_segments)} 个片段")
        print(f"\n训练集包含受试者：{', '.join(train_studies)}")
        print(f"测试集包含受试者：{', '.join(test_studies)}")
        
        # 统计标签分布
        print(f"\n训练集标签分布:")
        train_label_count = defaultdict(int)
        for seg in train_segments:
            train_label_count[seg['label']] += 1
        for label, count in sorted(train_label_count.items()):
            stage_name = [k for k, v in self.config['model_stage_mapping'].items() if v == label][0]
            print(f"  阶段 {stage_name} (标签{label}): {count}个 ({count/len(train_segments)*100:.1f}%)")
        
        print(f"\n测试集标签分布:")
        test_label_count = defaultdict(int)
        for seg in test_segments:
            test_label_count[seg['label']] += 1
        for label, count in sorted(test_label_count.items()):
            stage_name = [k for k, v in self.config['model_stage_mapping'].items() if v == label][0]
            print(f"  阶段 {stage_name} (标签{label}): {count}个 ({count/len(test_segments)*100:.1f}%)")
        
        # 4. 保存划分后的数据集（可选）
        if save_npy:
            os.makedirs(save_path, exist_ok=True)
            # 保存训练集
            train_save_dir = os.path.join(save_path, "train")
            os.makedirs(train_save_dir, exist_ok=True)
            np.save(os.path.join(train_save_dir, "all_segments.npy"), train_segments)
            # 保存测试集
            test_save_dir = os.path.join(save_path, "test")
            os.makedirs(test_save_dir, exist_ok=True)
            np.save(os.path.join(test_save_dir, "all_segments.npy"), test_segments)
            print(f"\n✅ 训练集保存至: {os.path.join(train_save_dir, 'all_segments.npy')}")
            print(f"✅ 测试集保存至: {os.path.join(test_save_dir, 'all_segments.npy')}")
        
        # 5. 返回PyTorch数据集
        train_dataset = SleepStagingDataset(train_segments, self.config)
        test_dataset = SleepStagingDataset(test_segments, self.config)
        return train_dataset, test_dataset

def load_saved_dataset(npy_path: str, config: Dict[str, Any]) -> Dataset:
    """从保存的.npy文件加载数据集"""
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"未找到保存的数据集文件: {npy_path}")
    
    print(f"从 {npy_path} 加载数据集...")
    all_segments = np.load(npy_path, allow_pickle=True).tolist()
    print(f"成功加载 {len(all_segments)} 个片段（仅EEG C3-A2和EOG LOC-A2，标签来自UserStaging）")
    
    # 统计标签分布
    label_count = defaultdict(int)
    for seg in all_segments:
        label_count[seg['label']] += 1
    print("标签分布:")
    for label, count in sorted(label_count.items()):
        stage_name = [k for k, v in config['model_stage_mapping'].items() if v == label][0]
        print(f"  阶段 {stage_name} (标签{label}): {count}个片段 ({count/len(all_segments)*100:.1f}%)")
    
    return SleepStagingDataset(all_segments, config)

def get_loader_from_saved(
    npy_path: str,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """从保存的.npy文件创建数据加载器"""
    dataset = load_saved_dataset(npy_path, CONFIG)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
