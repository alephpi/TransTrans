# 转录转录
将 B 站长视频音频转录为文本，自用为主

# 工作栈

- [Yutto](https://github.com/yutto-dev/yutto)：下载音频
- ffmpeg：音频转码
- [FunASR](https://github.com/modelscope/FunASR)：ASR

# 环境配置
本仓库用 `uv` 管理依赖。~~不用 `uv` 的，我想你们肯定知道怎么做。~~

## 仅使用
```sh
git clone https://github.com/alephpi/TransTrans.git
cd TransTrans
uv venv
uv sync --production
```

## 开发
```sh
git clone https://github.com/alephpi/TransTrans.git
cd TransTrans
uv venv
uv sync
```

# 使用方式

## 示例
以 BV1iddQYQE7D 为例，运行
```sh
source .venv/bin/activate
python main.py BV1iddQYQE7D -w hotwords.txt
```
或者
```sh
uv python main.py BV1iddQYQE7D -w hotwords.txt
```

这将首先调用 `yutto` 下载音频，然后经过 `ffmpeg` 转码，然后用 `FunASR` 转录得到文本（有带时间戳字幕和纯文本两种），期间处理结果均保存于 `data/{bvid}` 文件夹下。

## 自定义热词
可在 `hotwords.txt` 中逐行添加音频中出现的高频词汇，以提高识别准确率。

# 配置要求
- 系统自行安装 ffmpeg（windows 注意配置其安装路径到环境变量）
- GPU 显存不少于 4G，如果运行时超出，适当缩小 `batch_size_s` 即可，`batch_size_s=300`时，显存占用约 2G。

# 笔记

## 模型选型
根据 [论文](https://arxiv.org/pdf/2407.04051) 表格 6，Paraformer-zh 在 CER 和 RTF 上都达到最好。故不考虑 SenseVoice-small。

## 转录（`transcript.py`）
无论原音频编码如何，在 FunASR 中都以`torchaudio`导入并重采样至 16khz 处理。

仍以 BV1iddQYQE7D 为例（时长两小时），yutto 提供三种码率的`m4a`音频，编码均为 `aac`，其属性值如下（`torchaudio.info`）

|  码率   | 大小 | 采样率 | 帧数 |
| :-----: | :--: | :----: | :--: |
| 64kbps  | 35M  | 48khz  | 180k |
| 128kbps | 77M  | 48khz  | 360k |
| 320kbps | 138M | 48khz  | 360k |

对这些不同码率的音频，无论是 `torchaudio` 内部的重采样还是用 `ffmpeg` 的重采样，都得到 235M 的 `wav` 文件，且对识别结果无显著影响。

平均转录时长/音频时长比例（rtf_avg）为 0.008，即两小时音频一分钟转录完毕。

funasr AutoModel 在 `punc_model` 空置时， `sentence_timestamp=True` 时报错。

## 清理口语化表达（`deoral.py`）

口语化特征：
1. 填充词 filler：就是、什么的、他妈的
2. 重复、修正 repetition
3. 语气词：哎呀
4. 成分省略

# Experiments

# TODO
- [x] 文本转录
- [ ] 清理口语化表达
- [ ] 自动切片
