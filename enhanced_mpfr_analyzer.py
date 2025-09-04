#!/usr/bin/env python3
"""
增强版自动化浮点误差分析工具
基于对JSON案例的深度分析，优化函数分类和测试模板
支持多种数值计算函数类型的专业化测试
"""

import json
import re
import os
import subprocess
import sys
from typing import List, Dict, Tuple, Optional
import tempfile
import shutil

# 导入测试模板生成器
from test_templates import TestTemplateGenerator

class EnhancedMPFRAnalyzer:
    def __init__(self, input_file: str, output_dir: str, mpfr_precision: int = 128):
        self.input_file = input_file
        self.output_dir = output_dir
        self.mpfr_precision = mpfr_precision
        self.cases = []
        self.all_errors = []
        self.function_stats = {}

        # 初始化测试模板生成器
        self.template_generator = TestTemplateGenerator(mpfr_precision)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def load_cases(self):
        """加载JSON案例文件"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # JSON解析逻辑
            json_objects = []
            current_obj = ""
            brace_count = 0
            in_string = False
            escape_next = False

            for char in content:
                if escape_next:
                    current_obj += char
                    escape_next = False
                    continue

                if char == '\\':
                    current_obj += char
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string

                current_obj += char

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                        if brace_count == 0 and current_obj.strip():
                            try:
                                obj = json.loads(current_obj.strip())
                                json_objects.append(obj)
                                current_obj = ""
                            except json.JSONDecodeError:
                                pass

            if current_obj.strip():
                try:
                    obj = json.loads(current_obj.strip())
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    pass

            self.cases = json_objects
            print(f"成功加载 {len(self.cases)} 个案例")

        except Exception as e:
            print(f"加载案例文件失败: {e}")
            sys.exit(1)

    def extract_input_range(self, content: str) -> Optional[str]:
        """提取输入范围"""
        patterns = [
            r'x∈\[(.*?)\]',
            r'输入范围[：:]\s*.*?x∈\[(.*?)\]',
            r'principal∈\[(.*?)\]',
            r'rate∈\[(.*?)\]',
            r'输入范围.*?：(.*?)(?=。|$)',
            r'Input range.*?:(.*?)(?=\.|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "未指定范围"

    def extract_test_samples(self, content: str) -> List[float]:
        """提取测试样本"""
        samples = []

        # 科学计数法格式
        scientific_pattern = r'([+-]?[0-9]*\.?[0-9]+[eE][+-]?[0-9]+)'
        matches = re.findall(scientific_pattern, content)

        for match in matches:
            try:
                val = float(match)
                if abs(val) < 1e20:  # 过滤过大值
                    samples.append(val)
            except ValueError:
                continue

        # 普通数值格式
        normal_pattern = r'x=([+-]?[0-9]*\.?[0-9]+)'
        matches = re.findall(normal_pattern, content)

        for match in matches:
            try:
                val = float(match)
                if abs(val) < 1e20:
                    samples.append(val)
            except ValueError:
                continue

        # 去重并限制数量
        samples = list(set(samples))
        samples.sort(key=lambda x: abs(x))

        # 如果没有找到样本，根据函数类型添加默认样本
        if not samples:
            if 'sin' in content.lower():
                samples = [1e-10, -1e-10, 1e-8, -1e-8, 0.0, -0.0]
            elif 'exp' in content.lower():
                samples = [1e-12, -1e-12, 1e-8, -1e-8, 0.0]
            else:
                samples = [1e-10, -1e-10, 1e-8, -1e-8, 0.0]

        return samples[:10]

    def determine_function_category(self, content: str) -> Dict[str, str]:
        """增强的函数分类，返回主类别和子类别"""
        content_lower = content.lower()

        # 基础数学函数类 (transcendental)
        if any(func in content_lower for func in ['exp_minus_one', 'expm1']):
            return {'category': 'transcendental', 'subtype': 'expm1'}
        elif any(func in content_lower for func in ['optimized_sin', 'sin(']):
            return {'category': 'transcendental', 'subtype': 'sin'}
        elif any(func in content_lower for func in ['sinh_optimized', 'sinh']):
            return {'category': 'transcendental', 'subtype': 'sinh'}
        elif any(func in content_lower for func in ['tanh_optimized', 'tanh']):
            return {'category': 'transcendental', 'subtype': 'tanh'}
        elif any(func in content_lower for func in ['sqrt_optimized', 'sqrt']):
            return {'category': 'transcendental', 'subtype': 'sqrt'}
        elif any(func in content_lower for func in ['log1p_optimized', 'log1p']):
            return {'category': 'transcendental', 'subtype': 'log1p'}
        elif any(func in content_lower for func in ['high_precision_cos', 'cos']):
            return {'category': 'transcendental', 'subtype': 'cos'}

        # 金融计算类 (financial)
        elif any(func in content_lower for func in ['compound_value', '复利']):
            return {'category': 'financial', 'subtype': 'compound'}
        elif any(func in content_lower for func in ['implied_volatility', '隐含波动率']):
            return {'category': 'financial', 'subtype': 'volatility'}
        elif any(func in content_lower for func in ['internal_rate_of_return', 'irr']):
            return {'category': 'financial', 'subtype': 'irr'}
        elif any(func in content_lower for func in ['bond_duration', 'macaulay']):
            return {'category': 'financial', 'subtype': 'duration'}
        elif any(func in content_lower for func in ['monthly_payment', 'loan']):
            return {'category': 'financial', 'subtype': 'payment'}
        elif any(func in content_lower for func in ['value_at_risk', 'var']):
            return {'category': 'financial', 'subtype': 'var'}

        # 级数求和类 (series)
        elif any(func in content_lower for func in ['direct_sum', '级数']):
            return {'category': 'series', 'subtype': 'direct_sum'}
        elif any(func in content_lower for func in ['borel_sum', 'borel']):
            return {'category': 'series', 'subtype': 'borel_sum'}

        # 数值积分类 (integration)
        elif any(func in content_lower for func in ['adaptive_integrate', '积分']):
            return {'category': 'integration', 'subtype': 'adaptive'}
        elif any(func in content_lower for func in ['adaptive_singular_integral', '奇异积分']):
            return {'category': 'integration', 'subtype': 'singular'}

        # 优化问题类 (optimization)
        elif any(func in content_lower for func in ['global_min', '全局最小']):
            return {'category': 'optimization', 'subtype': 'global_min'}

        # 线性代数类 (linear_algebra)
        elif any(func in content_lower for func in ['extreme_eigenpair', '特征值', 'eigenvalue']):
            return {'category': 'linear_algebra', 'subtype': 'eigenvalue'}

        return {'category': 'unknown', 'subtype': 'generic'}

    def extract_user_code(self, output_content: str) -> Optional[str]:
        """从输出内容中提取用户代码"""
        patterns = [
            r'```c\n(.*?)\n```',
            r'```cpp\n(.*?)\n```',
            r'```\n(.*?)\n```',
        ]

        for pattern in patterns:
            match = re.search(pattern, output_content, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if 'double ' in code and '(' in code and ')' in code:
                    return code

        return None

    def generate_test_code(self, case_id: int, case_data: Dict, samples: List[float], func_info: Dict[str, str]) -> Optional[str]:
        """生成测试代码 - 增强版"""
        output_content = case_data['output']['content']
        target = case_data.get('target', '')
        user_code = self.extract_user_code(output_content)

        category = func_info['category']
        subtype = func_info['subtype']

        # 使用模板生成器生成测试代码
        return self.template_generator.generate_test_code(user_code, target, samples, category, subtype)




    def compile_and_run_test(self, case_id: int, test_code: str) -> Optional[List[Tuple[float, float, float]]]:
        """编译并运行测试代码"""
        if not test_code:
            return None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(test_code)
            c_file = f.name

        try:
            exe_file = c_file.replace('.c', '')

            compile_commands = [
                ['gcc', '-o', exe_file, c_file, '-I/opt/homebrew/include', '-L/opt/homebrew/lib', '-lmpfr', '-lgmp', '-lm'],
                ['gcc', '-o', exe_file, c_file, '-lmpfr', '-lgmp', '-lm'],
                ['gcc', '-o', exe_file, c_file, '-I/usr/local/include', '-L/usr/local/lib', '-lmpfr', '-lgmp', '-lm']
            ]

            compiled = False
            for cmd in compile_commands:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    compiled = True
                    break

            if not compiled:
                print(f"案例 {case_id+1} 编译失败")
                return None

            run_result = subprocess.run([exe_file], capture_output=True, text=True)
            if run_result.returncode != 0:
                print(f"案例 {case_id+1} 运行失败: {run_result.stderr}")
                return None

            # 解析结果
            errors = []
            for line in run_result.stdout.strip().split('\n'):
                if line.startswith('#'):
                    continue
                parts = line.split(', ')
                if len(parts) >= 4:
                    try:
                        x_val = float(parts[0])
                        user_val = float(parts[1])
                        rel_error = float(parts[3])
                        errors.append((x_val, user_val, rel_error))
                    except ValueError:
                        continue

            return errors

        except Exception as e:
            print(f"案例 {case_id+1} 处理异常: {e}")
            return None
        finally:
            try:
                os.unlink(c_file)
                if os.path.exists(exe_file):
                    os.unlink(exe_file)
            except:
                pass

    def process_all_cases(self):
        """处理所有案例"""
        for i, case_data in enumerate(self.cases):
            print(f"\n处理案例 {i+1}/{len(self.cases)}...")

            input_content = case_data['input'][0]['content']
            output_content = case_data['output']['content']

            input_range = self.extract_input_range(input_content)
            samples = self.extract_test_samples(input_content)
            func_info = self.determine_function_category(input_content)

            category = func_info['category']
            subtype = func_info['subtype']

            print(f"  函数类别: {category}")
            print(f"  函数子类: {subtype}")
            print(f"  输入范围: {input_range}")
            print(f"  测试样本: {samples[:5]}...")

            # 统计函数类型
            if category not in self.function_stats:
                self.function_stats[category] = []
            self.function_stats[category].append(i+1)

            # 生成测试代码
            test_code = self.generate_test_code(i, case_data, samples, func_info)
            if not test_code:
                print(f"  无法生成测试代码")
                continue

            # 编译运行
            errors = self.compile_and_run_test(i, test_code)
            if errors:
                print(f"  获得 {len(errors)} 个误差数据点")

                # 保存案例误差统计
                self.save_case_errors(i, errors, category, subtype, input_range)

                # 添加到全局误差列表
                for x_val, user_val, rel_error in errors:
                    self.all_errors.append((i, x_val, user_val, rel_error))
            else:
                print(f"  未获得有效误差数据")

    def save_case_errors(self, case_id: int, errors: List[Tuple[float, float, float]],
                        category: str, subtype: str, input_range: str):
        """保存单个案例的误差统计"""
        sorted_errors = sorted(errors, key=lambda x: x[2], reverse=True)[:5]

        output_file = os.path.join(self.output_dir, f"case{case_id+1}_error_stats.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 案例 {case_id+1} 误差统计\n\n")
            f.write(f"**函数类别**: {category}\n")
            f.write(f"**函数子类**: {subtype}\n")
            f.write(f"**输入范围**: {input_range}\n")
            f.write(f"**MPFR精度**: {self.mpfr_precision} 位\n\n")

            f.write("## 误差最大的5个样本\n\n")
            f.write("| x值 | 用户函数值 | 相对误差 |\n")
            f.write("|-----|-----------|----------|\n")

            for x_val, user_val, rel_error in sorted_errors:
                f.write(f"| {x_val:.6e} | {user_val:.6e} | {rel_error:.6e} |\n")

            f.write(f"\n**总测试样本数**: {len(errors)}\n")
            if errors:
                avg_error = sum(e[2] for e in errors) / len(errors)
                f.write(f"**平均相对误差**: {avg_error:.6e}\n")

    def generate_summary_report(self):
        """生成增强的汇总报告"""
        if not self.all_errors:
            print("没有误差数据，无法生成汇总报告")
            return

        sorted_errors = sorted(self.all_errors, key=lambda x: x[3], reverse=True)[:22]

        output_file = os.path.join(self.output_dir, "enhanced_summary_report.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 浮点数值计算代码生成汇总报告\n\n")
            f.write(f"**总案例数**: {len(self.cases)}\n")
            f.write(f"**MPFR精度**: {self.mpfr_precision} 位\n\n")

            # 函数类别分布统计
            f.write("## 函数类别分布\n\n")
            for category, case_list in self.function_stats.items():
                f.write(f"- **{category}**: {len(case_list)} 个案例 (case {', case '.join(map(str, case_list))})\n")

            f.write(f"\n## 误差最大的10个样本（所有案例）\n\n")
            f.write("| 案例ID | x值 | 用户函数值 | 相对误差 |\n")
            f.write("|--------|-----|-----------|----------|\n")

            for case_id, x_val, user_val, rel_error in sorted_errors:
                f.write(f"| case{case_id+1} | {x_val:.6e} | {user_val:.6e} | {rel_error:.6e} |\n")

            # 统计信息
            if self.all_errors:
                avg_error = sum(e[3] for e in self.all_errors) / len(self.all_errors)
                max_error = max(e[3] for e in self.all_errors)
                min_error = min(e[3] for e in self.all_errors if e[3] > 0)

                f.write(f"\n## 误差统计\n\n")
                f.write(f"- **最大相对误差**: {max_error:.6e}\n")
                f.write(f"- **最小相对误差**: {min_error:.6e}\n")
                f.write(f"- **平均相对误差**: {avg_error:.6e}\n")


def main():
    """主函数"""
    try:
        result = subprocess.run(['gcc', '--version'], capture_output=True)
        if result.returncode != 0:
            print("错误: 未找到gcc编译器")
            sys.exit(1)
    except:
        print("错误: 未找到gcc编译器")
        sys.exit(1)

    analyzer = EnhancedMPFRAnalyzer(
        input_file="模型代码生成浮点数值计算能力评测.json",
        output_dir="./enhanced_results/",
        mpfr_precision=128
    )

    print("=== 增强版MPFR浮点误差分析工具 ===")
    print(f"输入文件: {analyzer.input_file}")
    print(f"输出目录: {analyzer.output_dir}")
    print(f"MPFR精度: {analyzer.mpfr_precision} 位")

    analyzer.load_cases()
    analyzer.process_all_cases()
    analyzer.generate_summary_report()

    print(f"\n分析完成！结果保存在 {analyzer.output_dir}")


if __name__ == "__main__":
    main()
