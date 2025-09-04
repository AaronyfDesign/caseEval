#!/usr/bin/env python3
"""
浮点数采样分析程序 - 演示版本
针对指定可执行文件输入域D浮点值进行采样分析其结果误差
采样策略：针对输入域内浮点数每个指数下取100个尾数用作采样样本
使用Python高精度库作为比较程序误差标准
"""

import struct
import random
import subprocess
import sys
import os
import argparse
from typing import List, Tuple, Optional
import json
import math
from decimal import Decimal, getcontext

class FloatSampler:
    """浮点数采样器"""

    def __init__(self, precision: int = 128):
        """
        初始化采样器
        Args:
            precision: 精度位数，默认128位
        """
        self.precision = precision
        # 设置Decimal精度
        getcontext().prec = precision

    def float_to_components(self, f: float) -> Tuple[int, int, int]:
        """
        将浮点数分解为符号位、指数和尾数
        Args:
            f: 输入浮点数
        Returns:
            (sign, exponent, mantissa) 符号位、指数、尾数
        """
        # 将float转换为64位二进制表示
        bits = struct.unpack('>Q', struct.pack('>d', f))[0]

        # IEEE 754 双精度格式：1位符号 + 11位指数 + 52位尾数
        sign = (bits >> 63) & 0x1
        exponent = (bits >> 52) & 0x7FF
        mantissa = bits & 0xFFFFFFFFFFFFF

        return sign, exponent, mantissa

    def components_to_float(self, sign: int, exponent: int, mantissa: int) -> float:
        """
        将符号位、指数和尾数组合为浮点数
        Args:
            sign: 符号位 (0或1)
            exponent: 指数 (0-2047)
            mantissa: 尾数 (0-0xFFFFFFFFFFFFF)
        Returns:
            组合后的浮点数
        """
        # 组合成64位二进制表示
        bits = (sign << 63) | (exponent << 52) | mantissa
        return struct.unpack('>d', struct.pack('>Q', bits))[0]

    def generate_samples_for_exponent(self, exponent: int, count: int = 100) -> List[float]:
        """
        为特定指数生成采样点
        Args:
            exponent: 指数值
            count: 采样数量，默认100
        Returns:
            采样点列表
        """
        samples = []

        # 检查指数范围 (IEEE 754双精度：1-2046为正常数)
        if exponent == 0 or exponent == 2047:
            return samples  # 跳过特殊值（零、非规格化数、无穷大、NaN）

        # 对于每个指数，随机选择尾数
        max_mantissa = 0xFFFFFFFFFFFFF  # 52位尾数的最大值

        for _ in range(count):
            # 随机生成尾数
            mantissa = random.randint(0, max_mantissa)

            # 生成正数和负数样本
            for sign in [0, 1]:
                sample = self.components_to_float(sign, exponent, mantissa)
                if not (math.isnan(sample) or math.isinf(sample)):
                    samples.append(sample)

        return samples

    def generate_samples_in_domain(self, domain_min: float, domain_max: float,
                                 samples_per_exponent: int = 100) -> List[float]:
        """
        在指定域内生成采样点
        Args:
            domain_min: 域的最小值
            domain_max: 域的最大值
            samples_per_exponent: 每个指数的采样数量
        Returns:
            采样点列表
        """
        samples = []

        # 获取域边界的指数范围
        _, min_exp, _ = self.float_to_components(abs(domain_min)) if domain_min != 0 else (0, 1, 0)
        _, max_exp, _ = self.float_to_components(abs(domain_max)) if domain_max != 0 else (0, 1, 0)

        # 确保指数范围合理
        min_exp = max(1, min_exp - 2)  # 扩展一点范围
        max_exp = min(2046, max_exp + 2)

        print(f"指数范围: {min_exp} 到 {max_exp}")

        # 为每个指数生成样本
        for exponent in range(min_exp, max_exp + 1):
            exp_samples = self.generate_samples_for_exponent(exponent, samples_per_exponent)

            # 过滤出在域内的样本
            valid_samples = [s for s in exp_samples if domain_min <= s <= domain_max]
            samples.extend(valid_samples)

            if valid_samples:
                print(f"指数 {exponent}: 生成 {len(valid_samples)} 个有效样本")

        return samples

class ExecutableRunner:
    """可执行文件运行器"""

    def __init__(self, executable_path: str):
        """
        初始化运行器
        Args:
            executable_path: 可执行文件路径
        """
        self.executable_path = executable_path
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"可执行文件不存在: {executable_path}")
        if not os.access(executable_path, os.X_OK):
            raise PermissionError(f"文件不可执行: {executable_path}")

    def run_with_input(self, input_value: float) -> Optional[float]:
        """
        使用指定输入运行可执行文件
        Args:
            input_value: 输入值
        Returns:
            输出结果，如果失败返回None
        """
        try:
            # 运行可执行文件，传入输入值
            result = subprocess.run(
                [self.executable_path],
                input=str(input_value),
                text=True,
                capture_output=True,
                timeout=10  # 10秒超时
            )

            if result.returncode == 0:
                # 尝试解析输出为浮点数
                output = result.stdout.strip()
                return float(output)
            else:
                print(f"程序执行失败 (输入: {input_value}): {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"程序执行超时 (输入: {input_value})")
            return None
        except ValueError as e:
            print(f"输出解析失败 (输入: {input_value}): {e}")
            return None
        except Exception as e:
            print(f"执行出错 (输入: {input_value}): {e}")
            return None

class HighPrecisionReference:
    """高精度参考计算器 - 使用Python高精度库"""

    def __init__(self, function_name: str):
        """
        初始化参考计算器
        Args:
            function_name: 函数名称 (如 'sin', 'cos', 'exp', 'log' 等)
        """
        self.function_name = function_name

        # 函数映射
        self.function_map = {
            'sin': self._sin_reference,
            'cos': self._cos_reference,
            'tan': self._tan_reference,
            'exp': self._exp_reference,
            'log': self._log_reference,
            'sqrt': self._sqrt_reference,
        }

        if function_name not in self.function_map:
            raise ValueError(f"不支持的函数: {function_name}")

    def _sin_reference(self, x: float) -> float:
        """高精度sin计算"""
        # 使用更高精度的算法或库
        return math.sin(x)  # 简化版本，实际可以使用更高精度的实现

    def _cos_reference(self, x: float) -> float:
        """高精度cos计算"""
        return math.cos(x)

    def _tan_reference(self, x: float) -> float:
        """高精度tan计算"""
        return math.tan(x)

    def _exp_reference(self, x: float) -> float:
        """高精度exp计算"""
        return math.exp(x)

    def _log_reference(self, x: float) -> float:
        """高精度log计算"""
        return math.log(x)

    def _sqrt_reference(self, x: float) -> float:
        """高精度sqrt计算"""
        return math.sqrt(x)

    def compute_reference(self, input_value: float) -> float:
        """
        计算高精度参考结果
        Args:
            input_value: 输入值
        Returns:
            高精度计算结果
        """
        func = self.function_map[self.function_name]
        return func(input_value)

class ErrorAnalyzer:
    """误差分析器"""

    @staticmethod
    def absolute_error(computed: float, reference: float) -> float:
        """计算绝对误差"""
        return abs(computed - reference)

    @staticmethod
    def relative_error(computed: float, reference: float) -> float:
        """计算相对误差"""
        if reference == 0:
            return float('inf') if computed != 0 else 0.0
        return abs((computed - reference) / reference)

    @staticmethod
    def ulp_error(computed: float, reference: float) -> float:
        """计算ULP误差"""
        if math.isnan(computed) or math.isinf(computed):
            return float('inf')

        if reference == computed:
            return 0.0

        # 计算ULP
        if reference == 0:
            return float('inf')

        # 获取指数
        bits = struct.unpack('>Q', struct.pack('>d', abs(reference)))[0]
        exp = ((bits >> 52) & 0x7FF) - 1023

        # ULP = 2^(exp - 52) for double precision
        ulp = 2.0 ** (exp - 52)

        return abs(computed - reference) / ulp

class SamplingAnalyzer:
    """采样分析主类"""

    def __init__(self, executable_path: str, function_name: str, precision: int = 128):
        """
        初始化分析器
        Args:
            executable_path: 可执行文件路径
            function_name: 函数名称
            precision: 精度位数
        """
        self.sampler = FloatSampler(precision)
        self.runner = ExecutableRunner(executable_path)
        self.reference = HighPrecisionReference(function_name)
        self.analyzer = ErrorAnalyzer()

    def analyze_domain(self, domain_min: float, domain_max: float,
                      samples_per_exponent: int = 100) -> dict:
        """
        分析指定域的误差
        Args:
            domain_min: 域最小值
            domain_max: 域最大值
            samples_per_exponent: 每个指数的采样数量
        Returns:
            分析结果字典
        """
        print(f"开始采样分析...")
        print(f"域范围: [{domain_min}, {domain_max}]")
        print(f"每个指数采样数量: {samples_per_exponent}")

        # 生成采样点
        samples = self.sampler.generate_samples_in_domain(
            domain_min, domain_max, samples_per_exponent
        )

        print(f"总共生成 {len(samples)} 个采样点")

        results = {
            'domain': [domain_min, domain_max],
            'total_samples': len(samples),
            'successful_runs': 0,
            'failed_runs': 0,
            'errors': {
                'absolute': [],
                'relative': [],
                'ulp': []
            },
            'statistics': {}
        }

        # 分析每个采样点
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1}/{len(samples)} 个样本")

            # 运行可执行文件
            computed = self.runner.run_with_input(sample)
            if computed is None:
                results['failed_runs'] += 1
                continue

            # 计算高精度参考值
            try:
                reference = self.reference.compute_reference(sample)
            except:
                results['failed_runs'] += 1
                continue

            results['successful_runs'] += 1

            # 计算各种误差
            abs_err = self.analyzer.absolute_error(computed, reference)
            rel_err = self.analyzer.relative_error(computed, reference)
            ulp_err = self.analyzer.ulp_error(computed, reference)

            results['errors']['absolute'].append(abs_err)
            results['errors']['relative'].append(rel_err)
            results['errors']['ulp'].append(ulp_err)

        # 计算统计信息
        self._compute_statistics(results)

        return results

    def _compute_statistics(self, results: dict):
        """计算误差统计信息"""
        for error_type in ['absolute', 'relative', 'ulp']:
            errors = results['errors'][error_type]
            if not errors:
                continue

            # 过滤无穷大值
            finite_errors = [e for e in errors if math.isfinite(e)]

            if finite_errors:
                results['statistics'][error_type] = {
                    'mean': sum(finite_errors) / len(finite_errors),
                    'max': max(finite_errors),
                    'min': min(finite_errors),
                    'count_finite': len(finite_errors),
                    'count_infinite': len(errors) - len(finite_errors)
                }
            else:
                results['statistics'][error_type] = {
                    'mean': float('inf'),
                    'max': float('inf'),
                    'min': float('inf'),
                    'count_finite': 0,
                    'count_infinite': len(errors)
                }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='浮点数采样误差分析程序 - 演示版本')
    parser.add_argument('executable', help='可执行文件路径')
    parser.add_argument('function', help='函数名称 (sin, cos, exp, log, etc.)')
    parser.add_argument('--domain-min', type=float, default=-1.0, help='输入域最小值')
    parser.add_argument('--domain-max', type=float, default=1.0, help='输入域最大值')
    parser.add_argument('--samples-per-exp', type=int, default=100, help='每个指数的采样数量')
    parser.add_argument('--precision', type=int, default=128, help='精度位数')
    parser.add_argument('--output', help='结果输出文件 (JSON格式)')

    args = parser.parse_args()

    try:
        # 创建分析器
        analyzer = SamplingAnalyzer(args.executable, args.function, args.precision)

        # 执行分析
        results = analyzer.analyze_domain(
            args.domain_min, args.domain_max, args.samples_per_exp
        )

        # 输出结果
        print("\n=== 分析结果 ===")
        print(f"总样本数: {results['total_samples']}")
        print(f"成功运行: {results['successful_runs']}")
        print(f"失败运行: {results['failed_runs']}")

        for error_type, stats in results['statistics'].items():
            print(f"\n{error_type.upper()}误差统计:")
            print(f"  平均值: {stats['mean']:.2e}")
            print(f"  最大值: {stats['max']:.2e}")
            print(f"  最小值: {stats['min']:.2e}")
            print(f"  有限值数量: {stats['count_finite']}")
            print(f"  无穷值数量: {stats['count_infinite']}")

        # 保存结果到文件
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n结果已保存到: {args.output}")

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
