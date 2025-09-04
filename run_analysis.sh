#!/bin/bash

# 自动化浮点误差分析工具运行脚本
# 使用MPFR作为Oracle，统计相对浮点误差

echo "=== 自动化浮点误差分析工具 ==="
echo "检查环境..."

# 检查gcc编译器
if ! command -v gcc &> /dev/null; then
    echo "❌ 错误: 未找到gcc编译器"
    echo "请安装GCC编译器后重试"
    exit 1
fi

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到python3"
    echo "请安装Python 3后重试"
    exit 1
fi

# 检查MPFR库（尝试编译一个简单程序）
echo "检查MPFR库..."
cat > /tmp/test_mpfr.c << 'EOF'
#include <mpfr.h>
int main() {
    mpfr_t x;
    mpfr_init2(x, 128);
    mpfr_clear(x);
    return 0;
}
EOF

# 尝试不同的编译选项
MPFR_FOUND=false
for cmd in "gcc -o /tmp/test_mpfr /tmp/test_mpfr.c -I/opt/homebrew/include -L/opt/homebrew/lib -lmpfr -lgmp" \
           "gcc -o /tmp/test_mpfr /tmp/test_mpfr.c -lmpfr -lgmp" \
           "gcc -o /tmp/test_mpfr /tmp/test_mpfr.c -I/usr/local/include -L/usr/local/lib -lmpfr -lgmp"; do
    if $cmd 2>/dev/null; then
        MPFR_FOUND=true
        break
    fi
done

rm -f /tmp/test_mpfr.c /tmp/test_mpfr

if [ "$MPFR_FOUND" = false ]; then
    echo "❌ 错误: 未找到MPFR库"
    echo "请安装MPFR库后重试，例如："
    echo "  macOS: brew install mpfr"
    echo "  Ubuntu: sudo apt-get install libmpfr-dev"
    echo "  CentOS: sudo yum install mpfr-devel"
    exit 1
fi

echo "✅ 环境检查完成"

# 检查输入文件
if [ ! -f "代码生成浮点数值计算能力评测.json" ]; then
    echo "❌ 错误: 未找到输入文件 '代码生成浮点数值计算能力评测.json'"
    echo "请确保JSON文件在当前目录中"
    exit 1
fi

# 创建输出目录
mkdir -p results
echo "📁 输出目录已准备: ./results/"

# 运行分析工具
echo "🚀 开始分析..."
echo ""

python3 improved_mpfr_analyzer.py

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 分析完成！"
    echo ""
    echo "📊 生成的报告文件："
    ls -la results/*.md | while read line; do
        echo "  📄 $line"
    done
    echo ""
    echo "📋 主要报告："
    echo "  • summary_report.md - 综合汇总报告"
    echo "  • case*_error_stats.md - 各案例详细分析"
    echo ""
    echo "🔍 查看汇总结果："
    echo "  cat results/summary_report.md"
else
    echo "❌ 分析过程中出现错误"
    exit 1
fi
