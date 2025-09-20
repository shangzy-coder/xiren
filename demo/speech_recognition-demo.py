#!/usr/bin/env python3
"""
离线语音识别 - 使用 Sherpa-ONNX
功能：语音活动检测 + 批量识别 + GPU加速
"""

import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import sherpa_onnx
    import soundfile as sf
    import numpy as np
    import librosa
except ImportError as e:
    print(f"请安装依赖: pip install -r requirements.txt")
    sys.exit(1)


def load_audio(file_path):
    """加载音频文件，支持多种格式"""
    print(f"正在加载音频文件: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        # 首先尝试使用librosa加载，支持更多格式
        samples, sample_rate = librosa.load(file_path, sr=None, mono=True)
        print(f"使用librosa加载成功")
    except Exception as e:
        print(f"librosa加载失败: {e}，尝试使用soundfile")
        try:
            samples, sample_rate = sf.read(file_path, dtype='float32')
            # 转换为单声道
            if len(samples.shape) > 1:
                samples = np.mean(samples, axis=1)
        except Exception as e2:
            raise RuntimeError(f"无法加载音频文件 {file_path}: {e2}")
    
    # VAD需要16kHz采样率，如果原始采样率不是16kHz，需要重采样
    if sample_rate != 16000:
        print(f"原始采样率: {sample_rate}Hz, 重采样到16kHz")
        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    
    duration = len(samples) / sample_rate
    print(f"音频信息: 采样率={sample_rate}Hz, 时长={duration:.2f}秒")
    
    return samples, sample_rate


def create_recognizer(use_gpu=False, num_threads=4, language="auto"):
    """创建语音识别器"""
    print("正在初始化语音识别模型...")

    provider = "cuda" if use_gpu else "cpu"
    print(f"使用计算设备: {provider}")
    print(f"语言设置: {language}")

    # 验证语言参数
    valid_languages = ["auto", "zh", "en", "ja", "ko", "yue"]
    if language not in valid_languages:
        print(f"警告: 无效语言 '{language}'，使用 'auto' 自动检测")
        language = "auto"

    # 使用本地SenseVoice模型（支持中文、英文、日文、韩文、粤语）
    # 新版本2025-09-09包含完整功能：ASR + SER + AED + 标点
    # model_dir = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2025-09-09"
    model_dir = "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{model_dir}/model.onnx",
        tokens=f"{model_dir}/tokens.txt",
        provider=provider,
        num_threads=num_threads,
        language=language,
        debug=False,
    )

    print("语音识别模型初始化完成")
    return recognizer


def create_punctuation_processor(use_gpu=False, num_threads=2):
    """创建标点符号处理器"""
    print("正在初始化标点模型...")

    provider = "cuda" if use_gpu else "cpu"
    print(f"标点模型使用计算设备: {provider}")

    # 标点模型路径
    punct_model_dir = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"

    try:
        # 创建标点模型配置
        model_config = sherpa_onnx.OfflinePunctuationModelConfig()
        model_config.ct_transformer = f"{punct_model_dir}/model.onnx"
        model_config.num_threads = num_threads
        model_config.provider = provider

        # 创建标点配置
        punctuation_config = sherpa_onnx.OfflinePunctuationConfig()
        punctuation_config.model = model_config

        punctuation_processor = sherpa_onnx.OfflinePunctuation(punctuation_config)
        print("标点模型初始化完成")
        return punctuation_processor

    except Exception as e:
        print(f"标点模型初始化失败: {e}")
        print("将跳过标点处理")
        return None


def create_speaker_recognition(use_gpu=False, num_threads=2, speaker_db_file=None):
    """创建说话人识别系统"""
    print("正在初始化说话人识别系统...")

    provider = "cuda" if use_gpu else "cpu"
    print(f"说话人识别使用计算设备: {provider}")

    try:
        # 创建说话人嵌入提取器
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig()
        config.model = "models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
        config.provider = provider
        config.num_threads = num_threads

        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

        # 创建说话人管理器
        manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)

        print(f"说话人识别系统初始化完成 (嵌入维度: {extractor.dim})")
        return extractor, manager

    except Exception as e:
        print(f"说话人识别系统初始化失败: {e}")
        print("将跳过说话人识别")
        return None, None


def register_speaker_from_audio(speaker_manager, speaker_extractor, speaker_name, audio_file):
    """从音频文件注册说话人"""
    try:
        # 加载音频
        samples, sample_rate = librosa.load(audio_file, sr=None, mono=True)

        # 提取嵌入
        stream = speaker_extractor.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()

        embedding = speaker_extractor.compute(stream)
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        # 注册到管理器
        success = speaker_manager.add(speaker_name, embedding.tolist())

        if success:
            print(f"✅ 说话人 '{speaker_name}' 注册成功")
            return True
        else:
            print(f"❌ 说话人 '{speaker_name}' 注册失败")
            return False

    except Exception as e:
        print(f"❌ 注册说话人 '{speaker_name}' 时出错: {e}")
        return False


def detect_speech_segments(samples, sample_rate):
    """使用Sherpa-ONNX VAD进行语音段落检测"""
    print("正在进行语音段落检测...")
    
    # 创建VAD配置
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = "models/silero_vad.onnx"
    config.silero_vad.threshold = 0.5
    config.silero_vad.min_silence_duration = 0.25  # 秒
    config.silero_vad.min_speech_duration = 0.25   # 秒
    config.silero_vad.max_speech_duration = 5      # 秒
    config.sample_rate = sample_rate
    
    window_size = config.silero_vad.window_size
    
    # 创建VAD实例
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
    
    segments = []
    total_samples_processed = 0
    
    # 处理音频数据
    while len(samples) > total_samples_processed + window_size:
        chunk = samples[total_samples_processed:total_samples_processed + window_size]
        vad.accept_waveform(chunk)
        total_samples_processed += window_size
        
        # 获取检测到的语音段落
        while not vad.empty():
            segment_samples = vad.front.samples
            start_time = vad.front.start / sample_rate
            duration = len(segment_samples) / sample_rate
            end_time = start_time + duration

            segments.append({
                'samples': segment_samples,
                'sample_rate': sample_rate,
                'start_time': start_time,
                'end_time': end_time
            })
            vad.pop()
    
    # 处理剩余的音频数据
    vad.flush()
    while not vad.empty():
        segment_samples = vad.front.samples
        start_time = vad.front.start / sample_rate
        duration = len(segment_samples) / sample_rate
        end_time = start_time + duration

        segments.append({
            'samples': segment_samples,
            'sample_rate': sample_rate,
            'start_time': start_time,
            'end_time': end_time
        })
        vad.pop()
    
    print(f"检测到 {len(segments)} 个语音段落")
    return segments


def process_batch(recognizer, batch_segments, start_idx):
    """处理单个批次的语音段落"""
    # 创建识别流 - 预分配内存
    streams = [None] * len(batch_segments)

    for i, segment in enumerate(batch_segments):
        stream = recognizer.create_stream()
        # 直接传递numpy数组，避免不必要的复制
        stream.accept_waveform(segment['sample_rate'], segment['samples'])
        streams[i] = stream

    # 批量识别 - 这是真正的并行处理
    recognizer.decode_streams(streams)

    # 批量获取结果（包含完整功能：ASR + SER + AED）
    batch_results = []
    for i, stream in enumerate(streams):
        result = stream.result
        segment_idx = start_idx + i

        # 准备基础结果
        result_data = {
            'text': result.text,
            'emotion': result.emotion,      # 情感识别 (SER)
            'event': result.event,          # 音频事件检测 (AED)
            'language': result.lang,        # 语言识别
            'start_time': batch_segments[i]['start_time'],
            'end_time': batch_segments[i]['end_time'],
            'segment_idx': segment_idx,     # 保存原始索引用于排序
            'text_with_punct': result.text, # 默认为原始文本，如果标点处理失败
            'speaker': 'Unknown'           # 默认为未知说话人
        }

        batch_results.append(result_data)

    return batch_results


def add_punctuation_to_results(results, punctuation_processor):
    """为识别结果添加标点符号"""
    if not punctuation_processor:
        print("标点处理器不可用，跳过标点处理")
        return results

    print("正在为识别结果添加标点符号...")

    try:
        for result in results:
            if result['text'].strip():  # 只处理非空文本
                # 使用标点处理器添加标点
                punctuated_text = punctuation_processor.add_punctuation(result['text'])
                result['text_with_punct'] = punctuated_text
            else:
                result['text_with_punct'] = result['text']

        print(f"标点处理完成，已处理 {len(results)} 个段落")
        return results

    except Exception as e:
        print(f"标点处理过程中出错: {e}")
        print("将使用原始文本")
        return results


def add_speaker_recognition_to_results(results, speaker_extractor, speaker_manager, segments):
    """为识别结果添加说话人识别"""
    if not speaker_extractor or not speaker_manager:
        print("说话人识别系统不可用，跳过说话人识别")
        return results

    print("正在进行说话人识别...")

    try:
        # 临时说话人池子：存储本次分析中发现的新说话人嵌入
        temp_speakers = []  # [(speaker_id, embedding), ...]
        temp_speaker_count = 0

        for i, result in enumerate(results):
            # 获取对应的音频段落
            if i < len(segments):
                segment = segments[i]
                samples = segment['samples']
                sample_rate = segment['sample_rate']

                # 提取说话人嵌入
                stream = speaker_extractor.create_stream()
                stream.accept_waveform(sample_rate, samples)
                stream.input_finished()

                embedding = speaker_extractor.compute(stream)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)

                embedding_list = embedding.tolist()

                # 1. 首先在已注册的说话人池子中搜索
                registered_match = speaker_manager.search(embedding_list, threshold=0.3)  # 使用相似度阈值

                if registered_match:
                    # 找到匹配的已注册说话人
                    result['speaker'] = registered_match
                    continue

                # 2. 在临时说话人池子中搜索相似度匹配
                temp_match = None
                best_similarity = 0.0

                for temp_id, temp_embedding in temp_speakers:
                    # 计算余弦相似度
                    similarity = np.dot(embedding, temp_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(temp_embedding))
                    if similarity > 0.6 and similarity > best_similarity:  # 相似度阈值
                        temp_match = temp_id
                        best_similarity = similarity

                if temp_match:
                    # 找到匹配的临时说话人
                    result['speaker'] = temp_match
                else:
                    # 未找到匹配，创建新的临时说话人
                    temp_speaker_count += 1
                    new_speaker_id = f"Speaker{temp_speaker_count}"
                    result['speaker'] = new_speaker_id

                    # 将新说话人加入临时池子
                    temp_speakers.append((new_speaker_id, embedding))

        print(f"说话人识别完成，已处理 {len(results)} 个段落，发现 {temp_speaker_count} 个不同说话人")
        return results

    except Exception as e:
        print(f"说话人识别过程中出错: {e}")
        print("将使用默认说话人标签")
        # 出错时为所有结果设置默认标签
        for i, result in enumerate(results):
            result['speaker'] = f"Speaker{i+1}"
        return results


def recognize_segments(recognizer, segments, batch_size=None, max_workers=None, punctuation_processor=None, speaker_extractor=None, speaker_manager=None):
    """批量识别语音段落 - 并行优化版本"""
    print("正在进行语音识别...")

    if not segments:
        return []

    total_segments = len(segments)

    # 智能选择批次大小和worker数量
    if batch_size is None or max_workers is None:
        # 根据段落总数智能选择参数
        if total_segments <= 10:
            # 小量数据：直接处理，不分批
            batch_size = total_segments
            max_workers = 1
        elif total_segments <= 50:
            # 中等数据：小批次，适量并行
            batch_size = max(5, total_segments // 4)
            max_workers = min(4, total_segments // batch_size + 1)
        elif total_segments <= 200:
            # 大量数据：中等批次，中等并行
            batch_size = max(10, total_segments // 8)
            max_workers = min(8, total_segments // batch_size + 1)
        else:
            # 海量数据：大批量，高并行
            batch_size = max(20, total_segments // 16)
            max_workers = min(16, total_segments // batch_size + 1)

    print(f"使用 {max_workers} 个线程并行处理，批次大小: {batch_size}，总段落数: {total_segments}")

    # 创建批次任务
    batch_tasks = []
    for start_idx in range(0, total_segments, batch_size):
        end_idx = min(start_idx + batch_size, total_segments)
        current_batch = segments[start_idx:end_idx]
        batch_tasks.append((current_batch, start_idx))

    # 并行处理所有批次
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_batch = {
            executor.submit(process_batch, recognizer, batch, start_idx): (batch_idx, start_idx)
            for batch_idx, (batch, start_idx) in enumerate(batch_tasks)
        }

        # 收集结果
        completed = 0
        for future in as_completed(future_to_batch):
            batch_idx, start_idx = future_to_batch[future]
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                completed += 1
                print(f"完成批次 {completed}/{len(batch_tasks)} (段落 {start_idx+1}-{start_idx+len(batch_results)})")
            except Exception as exc:
                print(f'批次 {batch_idx} 处理失败: {exc}')

    # 按照时间顺序重新排序结果
    all_results.sort(key=lambda x: x['start_time'])

    # 移除辅助字段
    for result in all_results:
        del result['segment_idx']

    # 如果提供了标点处理器，为结果添加标点
    if punctuation_processor:
        all_results = add_punctuation_to_results(all_results, punctuation_processor)

    # 如果提供了说话人识别系统，为结果添加说话人信息
    if speaker_extractor and speaker_manager:
        all_results = add_speaker_recognition_to_results(all_results, speaker_extractor, speaker_manager, segments)

    print(f"语音识别完成，处理了 {len(all_results)} 个段落")
    return all_results


def print_results(results):
    """打印识别结果（包含情感分析）"""
    print("\n" + "="*80)
    print("🎭 SenseVoice完整功能识别结果 (2025-09-09)")
    print("="*80)

    # 统计信息
    emotion_stats = {}
    language_stats = {}
    event_stats = {}

    try:
        for i, result in enumerate(results, 1):
            # 统计信息
            emotion = result.get('emotion', '<|UNKNOWN|>').strip('<|>').lower()
            language = result.get('language', '<|UNKNOWN|>').strip('<|>')
            event = result.get('event', '<|UNKNOWN|>').strip('<|>')

            emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1
            language_stats[language] = language_stats.get(language, 0) + 1
            event_stats[event] = event_stats.get(event, 0) + 1

            # 显示结果
            print(f"[{i:02d}] {result['start_time']:.2f}s - {result['end_time']:.2f}s")
            print(f"     📝 原文: {result['text']}")
            if 'text_with_punct' in result and result['text_with_punct'] != result['text']:
                print(f"     ✏️  标点: {result['text_with_punct']}")
            print(f"     😊 情感: {result.get('emotion', 'N/A')}")
            print(f"     🎯 事件: {result.get('event', 'N/A')}")
            print(f"     🌍 语言: {result.get('language', 'N/A')}")
            print(f"     👤 说话人: {result.get('speaker', 'Unknown')}")
            print()

        # 显示统计信息
        print("-" * 80)
        print("📊 分析统计:")

        if emotion_stats:
            print("   情感分布:")
            for emotion, count in sorted(emotion_stats.items()):
                percentage = (count / len(results)) * 100
                emoji_map = {
                    'happy': '😊',
                    'sad': '😢',
                    'angry': '😠',
                    'neutral': '😐',
                    'fearful': '😨',
                    'disgusted': '🤢',
                    'surprised': '😲',
                    'unknown': '🤔'
                }
                emoji = emoji_map.get(emotion, '🤔')
                print(f"     {emoji} {emotion.capitalize()}: {count} ({percentage:.1f}%)")

        if language_stats:
            print("   语言分布:")
            for lang, count in sorted(language_stats.items()):
                percentage = (count / len(results)) * 100
                print(f"     🌍 {lang}: {count} ({percentage:.1f}%)")

        if event_stats:
            print("   事件分布:")
            for event, count in sorted(event_stats.items()):
                percentage = (count / len(results)) * 100
                print(f"     🎯 {event}: {count} ({percentage:.1f}%)")

    except BrokenPipeError:
        # 处理管道被关闭的情况（如使用head命令）
        # 这是正常行为，不需要报错
        pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="离线语音识别")
    parser.add_argument("input", help="输入音频文件路径")
    parser.add_argument("--use-gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--num-threads", type=int, default=4, help="CPU线程数")
    parser.add_argument("--batch-size", type=int, default=None, help="批量处理大小，默认为所有段落一起处理")
    parser.add_argument("--max-workers", type=int, default=None, help="最大并行工作线程数，默认为CPU核心数*2")
    parser.add_argument("--language", type=str, default="zh",
                       choices=["auto", "zh", "en", "ja", "ko", "yue"],
                       help="指定识别语言 (auto=自动检测, zh=中文, en=英文, ja=日文, ko=韩文, yue=粤语)")
    parser.add_argument("--enable-punctuation", action="store_true",
                       help="启用标点符号处理功能")
    parser.add_argument("--enable-speaker-id", action="store_true",
                       help="启用说话人识别功能")
    parser.add_argument("--register-speaker", nargs=2, metavar=('NAME', 'AUDIO_FILE'),
                       help="注册说话人：--register-speaker 张三 audio.wav")

    args = parser.parse_args()

    print("="*60)
    print("🎭 SenseVoice 完整功能语音识别系统")
    print("   支持：ASR + SER + AED + 标点符号 + 说话人识别")
    print("="*60)

    try:
        # 1. 加载音频
        samples, sample_rate = load_audio(args.input)

        # 2. 创建识别器
        recognizer = create_recognizer(args.use_gpu, args.num_threads, args.language)

        # 3. 创建标点处理器（如果启用）
        punctuation_processor = None
        if args.enable_punctuation:
            punctuation_processor = create_punctuation_processor(args.use_gpu)

        # 4. 创建说话人识别系统（如果启用或需要注册）
        speaker_extractor, speaker_manager = None, None
        if args.enable_speaker_id or args.register_speaker:
            speaker_extractor, speaker_manager = create_speaker_recognition(args.use_gpu)

            # 如果有注册说话人的请求
            if args.register_speaker:
                speaker_name, speaker_audio = args.register_speaker
                success = register_speaker_from_audio(speaker_manager, speaker_extractor, speaker_name, speaker_audio)
                if success:
                    print(f"🎯 说话人 '{speaker_name}' 已注册")
                else:
                    print(f"❌ 说话人 '{speaker_name}' 注册失败")

                # 如果只是注册，不继续进行识别
                if not args.enable_speaker_id:
                    return

        # 5. 检测语音段落
        segments = detect_speech_segments(samples, sample_rate)

        if not segments:
            print("未检测到语音内容")
            return

        # 6. 识别语音段落
        start_time = time.time()
        results = recognize_segments(recognizer, segments, args.batch_size, args.max_workers, punctuation_processor, speaker_extractor, speaker_manager)
        end_time = time.time()

        # 7. 打印结果
        print_results(results)

        # 8. 统计信息
        try:
            total_duration = len(samples) / sample_rate
            processing_time = end_time - start_time
            rtf = processing_time / total_duration

            print(f"🎯 处理统计:")
            print(f"   音频时长: {total_duration:.2f}秒")
            print(f"   处理时间: {processing_time:.2f}秒")
            print(f"   实时因子: {rtf:.2f}")
            print(f"   识别段落: {len(results)}个")
            if punctuation_processor:
                print(f"   标点处理: ✅ 已启用")
            else:
                print(f"   标点处理: ❌ 未启用")

            if speaker_extractor and speaker_manager:
                print(f"   说话人识别: ✅ 已启用")
            else:
                print(f"   说话人识别: ❌ 未启用")
        except BrokenPipeError:
            # 处理管道被关闭的情况
            pass

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
