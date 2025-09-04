# 浮点数采样分析器

本目录包含浮点数采样分析器的源代码和工具。

## 文件说明

### 核心文件
- `sampler.py` - 完整版采样分析器（使用MPFR插桩器作为oracle）
- `sampler_demo.py` - 演示版采样分析器（使用Python标准库作为参考）

## 功能特性

### 采样策略
- **IEEE 754基础**: 基于浮点数二进制表示的科学采样
- **指数覆盖**: 针对每个指数范围生成样本
- **尾数随机化**: 随机生成尾数确保覆盖度
- **域内筛选**: 只保留在指定输入域内的样本

### 误差分析
- **绝对误差**: |computed - reference|
- **相对误差**: |computed - reference| / |reference|
- **ULP误差**: 以最后一位有效数字为单位

## 使用方法

### 完整版（推荐）
```bash
# 分析可执行文件的sin函数误差
python3 sampler.py ../tests/test_program sin \
    --domain-min -3.14159 --domain-max 3.14159 \
    --samples-per-exp 100 \
    --precision 256 \
    --output results.json
```

### 演示版（无MPFR依赖）
```bash
# 使用Python标准库作为参考的简化版本
python3 sampler_demo.py ../tests/test_sin_simple sin \
    --domain-min -1.0 --domain-max 1.0 \
    --samples-per-exp 50 \
    --output demo_results.json
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `executable` | 目标可执行文件路径 | - |
| `function` | 数学函数名称 | - |
| `--domain-min` | 输入域最小值 | -1.0 |
| `--domain-max` | 输入域最大值 | 1.0 |
| `--samples-per-exp` | 每个指数的采样数量 | 100 |
| `--precision` | MPFR精度位数 | 256 |
| `--output` | 结果输出文件(JSON) | - |

## 支持的函数

- 三角函数: sin, cos, tan, asin, acos, atan
- 双曲函数: sinh, cosh, tanh
- 指数对数: exp, log, sqrt
- 其他函数: 可扩展支持

## 输出格式

分析结果以JSON格式输出，包含：
- 采样统计信息
- 误差数据数组
- 统计摘要（平均值、最大值、最小值等）

## 依赖要求

### 完整版
- Python 3.6+
- MPFR和GMP库（用于高精度oracle）
- 插桩器（../tests/cpp_mpfr_instrumenter）

### 演示版
- Python 3.6+（仅标准库）

## 注意事项

- 确保目标可执行文件支持标准输入/输出
- 采样数量越大，分析越准确但耗时越长
- 高精度设置会显著增加计算时间
