# 通用MPFR浮点误差分析工具

## 🎯 项目简介

这是一个通用的自动化浮点误差分析工具，能够解析JSON格式的代码生成评测文件，自动提取、编译、运行并分析浮点数值计算函数的精度表现。使用**MPFR库**作为高精度数值Oracle，提供可靠的误差分析结果。

## ✨ 主要特性

### 🔧 **智能化处理**
- **自动函数类型识别**：支持 `expm1`、`sin`、`compound`、`volatility`、`irr` 等多种函数类型
- **智能测试样本提取**：从自然语言描述中自动提取数值测试样本
- **代码自动提取**：从Markdown格式的输出中提取C/C++代码块

### 🎯 **高精度验证**
- **MPFR Oracle**：使用128位精度MPFR库作为数值参考标准
- **相对误差计算**：精确计算到机器精度级别（~10^-17）
- **多编译选项**：自动尝试不同的MPFR库链接路径

### 📊 **全面分析报告**
- **案例级报告**：每个函数的详细误差分析
- **汇总报告**：所有案例的综合统计分析
- **Markdown格式**：便于阅读和分享的格式化报告

### 🚀 **自动化流程**
- **一键运行**：提供Shell脚本，自动检查环境、编译运行
- **错误处理**：完善的异常处理和错误恢复机制
- **临时文件管理**：自动清理临时编译文件

## 📦 安装要求

### 系统要求
- **操作系统**：Linux、macOS、或类Unix系统
- **编译器**：GCC (支持C99标准)
- **Python**：Python 3.6+
- **Shell**：Bash

### 依赖库
```bash
# macOS (使用Homebrew)
brew install mpfr gmp

# Ubuntu/Debian
sudo apt-get install libmpfr-dev libgmp-dev

# CentOS/RHEL
sudo yum install mpfr-devel gmp-devel

# 或使用dnf (较新版本)
sudo dnf install mpfr-devel gmp-devel
```

## 🚀 快速开始

### 1. 准备输入文件
将JSON评测文件命名为 `代码生成浮点数值计算能力评测.json` 并放在项目目录中。

JSON文件格式示例：
```json
{
    "input": [
        {
            "role": "user",
            "content": "实现函数double exp_minus_one(double x)，输入范围x∈[-1e-8, 1e-8]，测试样本(x=5.3e-9, -7.1e-9)"
        }
    ],
    "target": "return expm1(x);",
    "output": {
        "role": "assistant",
        "content": "```c\ndouble exp_minus_one(double x) {\n    return expm1(x);\n}\n```"
    }
}
```

### 2. 运行分析

#### 方法一：使用自动化脚本（推荐）
```bash
./run_analysis.sh
```

#### 方法二：直接运行Python工具
```bash
python3 improved_mpfr_analyzer.py
```

### 3. 查看结果
分析完成后，结果将保存在 `./results/` 目录中：

```bash
# 查看汇总报告
cat results/summary_report.md

# 查看具体案例分析
cat results/case1_error_stats.md
cat results/case2_error_stats.md
```

## 📋 输出报告说明

### 汇总报告 (`summary_report.md`)
- **总体统计**：处理的案例数量、有效数据点数量
- **误差排行**：按相对误差大小排序的前10个样本
- **统计信息**：最大误差、最小误差、平均误差

### 案例报告 (`case*_error_stats.md`)
- **函数信息**：函数类型、输入范围、MPFR精度
- **误差详情**：误差最大的5个测试样本
- **统计摘要**：总样本数、平均相对误差

## 🔧 配置选项

### Python工具参数
可以通过修改 `improved_mpfr_analyzer.py` 中的参数来调整配置：

```python
analyzer = ImprovedMPFRAnalyzer(
    input_file="代码生成浮点数值计算能力评测.json",  # 输入文件路径
    output_dir="./results/",                        # 输出目录
    mpfr_precision=128                              # MPFR精度位数
)
```

### 支持的函数类型
工具能自动识别以下函数类型：
- `expm1` - 指数函数 exp(x)-1
- `sin` - 正弦函数优化版本
- `compound` - 复利计算函数
- `volatility` - 期权隐含波动率
- `irr` - 内部收益率计算
- `series` - 级数求和
- `optimization` - 全局最优化
- `integration` - 数值积分
- `eigenvalue` - 特征值计算
- `singular_integral` - 奇异积分

## 🛠️ 故障排除

### 常见问题

#### 1. MPFR库未找到
```bash
# 错误信息
❌ 错误: 未找到MPFR库

# 解决方案
# macOS
brew install mpfr gmp

# Ubuntu
sudo apt-get install libmpfr-dev libgmp-dev
```

#### 2. 编译失败
```bash
# 检查编译器
gcc --version

# 手动测试MPFR链接
gcc -o test test.c -lmpfr -lgmp -lm
```

#### 3. JSON文件格式错误
- 确保JSON文件为有效格式
- 检查是否包含必要字段：`input`、`output`
- 验证文件编码为UTF-8

#### 4. 无测试样本提取
- 检查输入内容是否包含数值样本
- 确保数值格式正确（如 `5.3e-9`）
- 查看控制台输出的调试信息

## 📈 性能特点

### 精度水平
- **MPFR精度**：128位十进制精度（约400位二进制）
- **误差检测**：能检测到 10^-17 级别的相对误差
- **数值稳定性**：使用高精度运算避免中间结果精度丢失

### 处理能力
- **函数类型**：支持10多种数值计算函数类型
- **样本数量**：每个案例可处理10+个测试样本
- **批量处理**：可同时处理多个JSON案例

### 执行效率
- **编译缓存**：避免重复编译相同代码
- **并行友好**：支持多案例并行处理（未来版本）
- **资源清理**：自动清理临时文件，避免磁盘占用

## 🔬 技术架构

### 核心组件
```
ImprovedMPFRAnalyzer
├── load_cases()           # JSON案例加载
├── extract_input_range()  # 输入范围提取
├── extract_test_samples() # 测试样本提取
├── determine_function_type() # 函数类型识别
├── extract_user_code()    # 用户代码提取
├── generate_test_code()   # 测试代码生成
├── compile_and_run_test() # 编译运行测试
└── generate_summary_report() # 汇总报告生成
```

### 测试代码模板
- **EXPM1模板**：基于 `mpfr_expm1()` 的高精度参考
- **Sin模板**：基于 `mpfr_sin()` 的三角函数测试
- **复利模板**：多参数组合的金融计算测试
- **通用模板**：适用于未知函数类型的基础测试

## 📝 扩展开发

### 添加新函数类型
1. 在 `determine_function_type()` 中添加识别逻辑
2. 实现对应的 `generate_*_test()` 方法
3. 在 `generate_test_code()` 中添加调用分支

### 自定义测试模板
```python
def generate_custom_test(self, user_code, target, samples):
    """生成自定义函数测试代码"""
    func_impl = user_code if user_code else f"// {target}"

    test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    // 自定义测试逻辑
    return 0;
}}'''
    return test_code
```

## 📄 许可证

MIT License - 欢迎使用、修改和分发

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进此工具！

### 开发环境设置
```bash
git clone <repository>
cd mpfr-error-analyzer
pip install -r requirements.txt  # 如果有Python依赖
```

### 测试
```bash
# 运行单元测试
python3 -m pytest tests/

# 运行集成测试
./test_all_cases.sh
```

---

**🎯 项目目标**：为数值计算代码生成提供可靠的精度验证工具，推动高质量数值算法的发展。
