#!/bin/bash

# 本地目录
SRC_DIR="/root/ComfyUI/custom_nodes/ComfyUI-WaaS-API"

# 目标服务器信息
REMOTE_PATH="/root/comfyui/ComfyUI/custom_nodes"  # 使用绝对路径
REMOTE_USER="root"
REMOTE_HOST="ec18a79f8b92479bbfd91a7aec6ee941.region1.waas.aigate.cc"
REMOTE_PORT="45098"
PASSWORD="84d6cbb1002241449183a5e6e698dc9f"

set -e  # 任何命令失败都退出脚本

echo "开始部署..."

# 创建临时目录用于打包
TEMP_DIR="/tmp/ComfyUI-WaaS-API-deploy"
mkdir -p ${TEMP_DIR}/ComfyUI-WaaS-API

# 复制文件到临时目录，排除不需要的文件
rsync -av --exclude='vue/' --exclude='svelte/' --exclude='.git/' --exclude='node_modules/' --exclude='.gitignore' "${SRC_DIR}/" "${TEMP_DIR}/ComfyUI-WaaS-API/"

# 打包压缩
cd ${TEMP_DIR}
tar -czf ComfyUI-WaaS-API.tar.gz ComfyUI-WaaS-API

# 如果需要密码认证，使用 sshpass（需先安装：apt-get install sshpass）
# 创建目标目录
sshpass -p "${PASSWORD}" ssh -o StrictHostKeyChecking=no -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}"

# 传输压缩包
sshpass -p "${PASSWORD}" scp -o StrictHostKeyChecking=no -P ${REMOTE_PORT} ComfyUI-WaaS-API.tar.gz ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

# 在远程服务器解压并清理
sshpass -p "${PASSWORD}" ssh -o StrictHostKeyChecking=no -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST} "cd ${REMOTE_PATH} && tar -xzf ComfyUI-WaaS-API.tar.gz && rm ComfyUI-WaaS-API.tar.gz"

# 清理临时文件
rm -rf ${TEMP_DIR}

echo "部署完成！"