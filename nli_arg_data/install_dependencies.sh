#!/bin/bash

echo "安装去重工具所需的依赖包..."

# 安装sentence-transformers和相关依赖
pip install sentence-transformers torch transformers

echo "依赖包安装完成!"
echo "现在可以运行 python MRDissimilarity.py 进行数据去重"
