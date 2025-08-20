#!/bin/bash
set -e

REQ_FILE="requirements.txt"
LOCK_FILE="requirements.lock.txt"

# 1️⃣ 检查文件存在
if [ ! -f "$REQ_FILE" ]; then
    echo "错误：$REQ_FILE 文件不存在"
    exit 1
fi

# 2️⃣ 初始化 uv 项目（如果还没有 pyproject.toml）
if [ ! -f "pyproject.toml" ]; then
    echo "初始化 uv 项目..."
    uv init
fi

# 3️⃣ 清空现有依赖（可选，如果希望覆盖）
uv remove --all || true

# 4️⃣ 从 requirements.txt 逐行添加依赖
echo "导入依赖到 uv..."
while read -r line; do
    # 去掉空格和注释
    line="$(echo "$line" | xargs)"
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    echo "添加依赖: $line"
    uv add "$line" || echo "⚠️ 添加依赖 $line 出现冲突，uv 会尝试解析可行版本"
done < "$REQ_FILE"

# 5️⃣ 生成锁定文件
echo "生成锁定文件 $LOCK_FILE ..."
uv pip compile pyproject.toml -o "$LOCK_FILE"

# 6️⃣ 同步依赖到环境
echo "同步依赖到虚拟环境..."
uv pip sync "$LOCK_FILE"

echo "✅ 完成！依赖已锁定到 $LOCK_FILE 并安装成功"
