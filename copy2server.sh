#!/bin/bash

# 本地目录
SRC_DIR="/home/workspace/yizhi/ComfyUI-WaaS-API"

# 目标服务器信息
REMOTE_PATH="~/comfyui/ComfyUI/custom_nodes"

REMOTE_USER="root"
REMOTE_HOST="7b8e51c6a80842ccb9827feac7bd59c8.region1.waas.aigate.cc"
REMOTE_PORT="42387"
PASSWORD="3f10b3f548264359b3d6ad24e36178cb"

echo "开始部署..."

# 创建临时目录用于打包
TEMP_DIR="/tmp/ComfyUI-WaaS-API-deploy"
mkdir -p ${TEMP_DIR}/ComfyUI-WaaS-API

# 复制文件到临时目录的comfyui-waas子目录，排除vue和svelte目录
rsync -av --exclude='vue/' --exclude='svelte/' --exclude='.git/' --exclude='node_modules/' --exclude='.gitignore' ${SRC_DIR}/ ${TEMP_DIR}/ComfyUI-WaaS-API/

# 打包压缩
cd ${TEMP_DIR}
tar -czf ComfyUI-WaaS-API.tar.gz ComfyUI-WaaS-API

# 创建目标目录（避免scp失败）
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"

# 传输压缩包到目标服务器
scp -P ${REMOTE_PORT} ComfyUI-WaaS-API.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

# 在目标服务器解压并清理
ssh -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_PATH} && tar -xzf ComfyUI-WaaS-API.tar.gz && rm ComfyUI-WaaS-API.tar.gz"

# 清理临时文件
rm -rf ${TEMP_DIR}

echo "部署完成！"