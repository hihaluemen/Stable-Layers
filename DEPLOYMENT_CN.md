# Stable-Layers 国内云 GPU 部署

本文用于在国内云 GPU 机器上部署 Stable-Layers HTTP 推理服务。推荐先用单卡 A100 80G 做验证；主项目服务器只调用 HTTP 接口，不安装模型。

## 1. 机器准备

建议：Ubuntu 22.04/24.04、NVIDIA Driver 550+、CUDA 12.x、A100 80G/H100/H200、系统盘至少 100 GB，模型缓存盘至少 80 GB。GPU 安全组只允许主服务内网 IP 访问 8080 端口。

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y git python3-venv git-lfs
git lfs install
```

## 2. 获取代码和安装依赖

```bash
git clone -b feature/http-inference-service \
  https://github.com/hihaluemen/Stable-Layers.git
cd Stable-Layers

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# 云厂商镜像通常已预装 torch/CUDA，先检查并保留可用环境。
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
PY

# 只补充推理包，不重复安装 torch。这里使用清华 PyPI 镜像。
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --no-cache-dir \
  -r requirements-server.txt
python -m pip install -r requirements-server.txt
```

如果检查发现 torch 缺失或 CUDA 不可用，才按云厂商说明安装匹配驱动的版本，例如 torch 2.12 + CUDA 13.0。已有可用 torch 时不要重复安装其他 CUDA 轮子。

## 3. 国内 Hugging Face 镜像和模型下载

模型不要在第一次接口请求时下载。先配置缓存并下载基础模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export MODEL_CACHE_DIR="$PWD/models/huggingface"
export HF_HOME="$MODEL_CACHE_DIR"
export HF_HUB_CACHE="$MODEL_CACHE_DIR/hub"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER
export TRANSFORMERS_CACHE="$MODEL_CACHE_DIR/transformers"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    'Qwen/Qwen-Image-Layered',
    cache_dir=os.environ['HF_HUB_CACHE'],
))
PY
```

下载 Stable-Layers LoRA。压缩包部署时不会包含大模型权重和 LoRA，需要单独下载到当前目录的 `model/`；如果你是通过 Git LFS 完整拉取 fork，也可以跳过下载并直接检查文件。

```bash
mkdir -p "$PWD/model"
python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id='StabilityLabs/Stable-Layers',
    local_dir='./model',
    local_dir_use_symlinks=False,
)
print(path)
PY
```

检查 LoRA 文件（必须成功后再运行推理）：

如果出现 `cas-server.xethub.hf.co`、`401 Unauthorized` 或 `File reconstruction error`，通常是 Xet 网络/认证失败，不是磁盘已满。保留缓存目录后重新运行下载命令即可续传：

```bash
df -h .
du -sh "$MODEL_CACHE_DIR"
```

```bash
test -f model/adapter_config.json
test -f model/adapter_model.safetensors
sha256sum model/adapter_config.json model/adapter_model.safetensors
```

如果检查失败，查找实际下载位置：

```bash
find model -maxdepth 3 -type f \( -name adapter_config.json -o -name adapter_model.safetensors \) -print
```

如果镜像不稳定，可以在网络较好的机器下载 `/data/huggingface` 后用 `rsync --partial --progress` 传到 GPU 主机。不要把 Hugging Face token、模型权重或 `.env` 提交到 Git。

## 4. 先跑离线测试

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface
export STABLE_LAYERS_LORA=$PWD/model

python decompose.py \
  --input 1.png \
  --output "$PWD/stable-layers-output" \
  --lora "$PWD/model" \
  --base-model Qwen/Qwen-Image-Layered \
  --steps 50 \
  --guidance-scale 1.0 \
  --num-layers 4 \
  --size 640 \
  --transparent
```

检查 `layer_*.png` 是 RGBA，且 `composite.png` 与原图主体位置一致。正式接口必须保留 `--transparent`，否则输出是白底 RGB，不适合坐标恢复。

## 5. 启动 HTTP 服务

```bash
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/huggingface
export STABLE_LAYERS_LORA=$PWD/model
export STABLE_LAYERS_TIMEOUT_SECONDS=1800
export STABLE_LAYERS_ALPHA_THRESHOLD=16

uvicorn serve:app --host 0.0.0.0 --port 8080 --workers 1
```

验证：

```bash
curl http://127.0.0.1:8080/health
curl -X POST http://127.0.0.1:8080/v1/layer-decomposition \
  -F image=@/path/to/source.png \
  -o response.json
```

服务返回 RGBA 图层的 base64、alpha bbox、图层顺序和坐标画布尺寸。每张 GPU 只运行一个 worker；当前服务复用 `decompose.py`，每次请求会重新加载模型，适合验证和低频请求。高并发前应改成长驻模型进程。

服务默认把 alpha 小于 16 的近透明扩散噪声清零后再计算 bbox。若模型输出边缘仍有噪声，可将 `STABLE_LAYERS_ALPHA_THRESHOLD` 调高到 24 或 32；如果主体被误删，再调低到 8。

## 6. 对接主项目

主项目使用 `feature/stable-layers-provider` 分支。主服务 `.env.local` 配置：

```dotenv
LAYER_PROVIDER=stable_layers
STABLE_LAYERS_ENDPOINT=http://GPU内网IP:8080/v1/layer-decomposition
STABLE_LAYERS_MODEL=Stable-Layers
STABLE_LAYERS_TIMEOUT_SECONDS=1800
```

切换前必须用真实图片对账：图层数量、坐标恢复、珍珠数量、珠径、碰撞数量和导出文件。出现问题时将 `LAYER_PROVIDER` 改回 `seedream` 即可恢复旧链路。

## 7. 常见问题

```bash
nvidia-smi
du -sh /data/huggingface
file /data/stable-layers-output/*/layer_*.png
```

显存不足时先确认没有多个推理进程，并保持单 worker。不要先降低官方推荐的 50 steps 或提高 CFG；这些参数会明显影响分层质量。商业使用还需遵守 `LICENSE.md` 中的 Stability AI Community License 条款。
