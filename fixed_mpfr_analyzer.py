#!/usr/bin/env python3
"""
修复版自动化浮点误差分析工具
使用MPFR作为Oracle，统计相对浮点误差
支持多种函数类型的通用测试
修复了JSON解析问题以正确处理多个案例
"""

import json
import re
import os
import subprocess
import sys
from typing import List, Dict, Tuple, Optional
import tempfile
import shutil

class FixedMPFRAnalyzer:
    def __init__(self, input_file: str, output_dir: str, mpfr_precision: int = 128):
        self.input_file = input_file
        self.output_dir = output_dir
        self.mpfr_precision = mpfr_precision
        self.cases = []
        self.all_errors = []

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def load_cases(self):
        """加载JSON案例文件 - 修复版解析逻辑"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 修复：正确解析多个连续的JSON对象
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

                        # 当大括号平衡时，尝试解析JSON对象
                        if brace_count == 0 and current_obj.strip():
                            try:
                                obj = json.loads(current_obj.strip())
                                json_objects.append(obj)
                                current_obj = ""
                            except json.JSONDecodeError as e:
                                # 如果解析失败，继续累积内容
                                pass

            # 处理剩余的内容
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
            r'x\s*∈\s*\[(.*?)\]',
            r'输入范围[：:]\s*.*?x∈\[(.*?)\]',
            r'principal∈\[(.*?)\]',
            r'rate∈\[(.*?)\]',
            r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?\s*,\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        return "unknown"

    def extract_test_samples(self, content: str) -> List[float]:
        """提取测试样本"""
        samples = []

        # 寻找测试样本的模式
        patterns = [
            r'x=([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)',  # x=5e-8格式
            r'([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+))',     # 数值格式
        ]

        # 提取所有数值
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    val = float(match)
                    if abs(val) < 1e10:  # 过滤过大的数值
                        samples.append(val)
                except ValueError:
                    continue

        # 去重并排序
        samples = list(set(samples))
        samples.sort(key=lambda x: abs(x))

        # 如果没有找到样本，添加默认测试样本
        if not samples:
            samples = [1e-10, -1e-10, 1e-8, -1e-8, 0.0]

        # 限制样本数量
        return samples[:22]

    def determine_function_type(self, content: str) -> str:
        """确定函数类型"""
        content_lower = content.lower()

        if 'exp_minus_one' in content_lower or 'expm1' in content_lower:
            return 'expm1'
        elif 'optimized_sin' in content_lower or 'sin(' in content_lower:
            return 'sin'
        elif 'compound_value' in content_lower or '复利' in content:
            return 'compound'
        elif 'implied_volatility' in content_lower or '隐含波动率' in content:
            return 'volatility'
        elif 'internal_rate_of_return' in content_lower or 'irr' in content_lower:
            return 'irr'
        elif 'direct_sum' in content_lower or '级数' in content:
            return 'series'
        elif 'global_min' in content_lower or '全局' in content:
            return 'optimization'
        elif 'adaptive_integrate' in content_lower or '积分' in content:
            return 'integration'
        elif 'extreme_eigenpair' in content_lower or '特征值' in content:
            return 'eigenvalue'
        elif 'adaptive_singular_integral' in content_lower:
            return 'singular_integral'
        else:
            return 'unknown'

    def extract_user_code(self, output_content: str) -> Optional[str]:
        """从输出内容中提取用户代码"""
        # 寻找C代码块
        patterns = [
            r'```c\n(.*?)\n```',
            r'```cpp\n(.*?)\n```',
            r'```\n(.*?)\n```',
        ]

        for pattern in patterns:
            match = re.search(pattern, output_content, re.DOTALL)
            if match:
                code = match.group(1).strip()
                # 检查是否包含函数定义
                if 'double ' in code and '(' in code and ')' in code:
                    return code

        # 如果没找到代码块，尝试从target字段提取简单实现
        return None

    def generate_test_code(self, case_id: int, case_data: Dict, samples: List[float], func_type: str) -> Optional[str]:
        """生成测试代码"""
        output_content = case_data['output']['content']
        target = case_data.get('target', '')

        # 提取用户代码
        user_code = self.extract_user_code(output_content)

        # 根据函数类型生成测试代码
        if func_type == 'expm1':
            return self.generate_expm1_test(user_code, target, samples)
        elif func_type == 'sin':
            return self.generate_sin_test(user_code, target, samples)
        elif func_type == 'compound':
            return self.generate_compound_test(user_code, target)
        else:
            # 通用测试生成
            return self.generate_generic_test(user_code, target, samples, func_type)

    def generate_expm1_test(self, user_code: Optional[str], target: str, samples: List[float]) -> str:
        """生成EXPM1测试代码"""
        # 使用target或默认实现
        func_impl = user_code if user_code else f"""
double exp_minus_one(double x) {{
    return expm1(x);
}}"""

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    mpfr_t x, ref, user_result;
    mpfr_init2(x, {self.mpfr_precision});
    mpfr_init2(ref, {self.mpfr_precision});
    mpfr_init2(user_result, {self.mpfr_precision});

    double test_samples[] = {{{', '.join(map(str, samples[:5]))}}};
    int n_samples = {min(5, len(samples))};

    printf("# x_value, user_value, mpfr_value, relative_error\\n");

    for (int i = 0; i < n_samples; i++) {{
        double x_val = test_samples[i];

        mpfr_set_d(x, x_val, MPFR_RNDN);
        mpfr_expm1(ref, x, MPFR_RNDN);

        double user_val = exp_minus_one(x_val);
        mpfr_set_d(user_result, user_val, MPFR_RNDN);

        mpfr_t error, abs_ref;
        mpfr_init2(error, {self.mpfr_precision});
        mpfr_init2(abs_ref, {self.mpfr_precision});

        mpfr_sub(error, user_result, ref, MPFR_RNDN);
        mpfr_abs(abs_ref, ref, MPFR_RNDN);

        if (mpfr_cmp_d(abs_ref, 1e-30) > 0) {{
            mpfr_div(error, error, abs_ref, MPFR_RNDN);
            mpfr_abs(error, error, MPFR_RNDN);
        }} else {{
            mpfr_abs(error, error, MPFR_RNDN);
        }}

        printf("%.15e, %.15e, %.15e, %.15e\\n",
               x_val, user_val, mpfr_get_d(ref, MPFR_RNDN), mpfr_get_d(error, MPFR_RNDN));

        mpfr_clear(error);
        mpfr_clear(abs_ref);
    }}

    mpfr_clear(x);
    mpfr_clear(ref);
    mpfr_clear(user_result);
    return 0;
}}'''
        return test_code

    def generate_sin_test(self, user_code: Optional[str], target: str, samples: List[float]) -> str:
        """生成Sin测试代码"""
        func_impl = user_code if user_code else f"""
double optimized_sin(double x) {{
    if (isnan(x)) return x;
    if (isinf(x)) return NAN;
    if (x == 0.0) return x;
    if (fabs(x) < 1e-5) {{
        double x2 = x * x;
        return x * (1.0 - x2/6.0 * (1.0 - x2/20.0));
    }}
    return sin(x);
}}"""

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    mpfr_t x, ref, user_result;
    mpfr_init2(x, {self.mpfr_precision});
    mpfr_init2(ref, {self.mpfr_precision});
    mpfr_init2(user_result, {self.mpfr_precision});

    double test_samples[] = {{{', '.join(map(str, samples[:5]))}, 0.0, -0.0}};
    int n_samples = {min(5, len(samples)) + 2};

    printf("# x_value, user_value, mpfr_value, relative_error\\n");

    for (int i = 0; i < n_samples; i++) {{
        double x_val = test_samples[i];

        mpfr_set_d(x, x_val, MPFR_RNDN);
        mpfr_sin(ref, x, MPFR_RNDN);

        double user_val = optimized_sin(x_val);
        mpfr_set_d(user_result, user_val, MPFR_RNDN);

        mpfr_t error, abs_ref;
        mpfr_init2(error, {self.mpfr_precision});
        mpfr_init2(abs_ref, {self.mpfr_precision});

        mpfr_sub(error, user_result, ref, MPFR_RNDN);
        mpfr_abs(abs_ref, ref, MPFR_RNDN);

        if (mpfr_cmp_d(abs_ref, 1e-30) > 0) {{
            mpfr_div(error, error, abs_ref, MPFR_RNDN);
            mpfr_abs(error, error, MPFR_RNDN);
        }} else {{
            mpfr_abs(error, error, MPFR_RNDN);
        }}

        printf("%.15e, %.15e, %.15e, %.15e\\n",
               x_val, user_val, mpfr_get_d(ref, MPFR_RNDN), mpfr_get_d(error, MPFR_RNDN));

        mpfr_clear(error);
        mpfr_clear(abs_ref);
    }}

    mpfr_clear(x);
    mpfr_clear(ref);
    mpfr_clear(user_result);
    return 0;
}}'''
        return test_code

    def generate_compound_test(self, user_code: Optional[str], target: str) -> str:
        """生成复利测试代码"""
        func_impl = user_code if user_code else f"""
double compound_value(double principal, double rate, int years) {{
    if (rate == 0.0 || years == 0) return principal;
    if (rate < 1e-5) {{
        double log_base = years * rate * (1 - rate/2);
        return principal * exp(log_base);
    }}
    return principal * exp(years * log(1.0 + rate));
}}"""

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    double principals[] = {{1000.0, 10000.0, 100.0}};
    double rates[] = {{0.05, 0.001, 1e-6, 0.0}};
    int years_list[] = {{10, 50, 1}};

    printf("# principal, rate, years, user_value, mpfr_value, relative_error\\n");

    for (int p = 0; p < 3; p++) {{
        for (int r = 0; r < 4; r++) {{
            for (int y = 0; y < 3; y++) {{
                double principal = principals[p];
                double rate = rates[r];
                int years = years_list[y];

                double user_val = compound_value(principal, rate, years);

                mpfr_t mp_principal, mp_rate, mp_years, mp_result, mp_base;
                mpfr_init2(mp_principal, {self.mpfr_precision});
                mpfr_init2(mp_rate, {self.mpfr_precision});
                mpfr_init2(mp_years, {self.mpfr_precision});
                mpfr_init2(mp_result, {self.mpfr_precision});
                mpfr_init2(mp_base, {self.mpfr_precision});

                mpfr_set_d(mp_principal, principal, MPFR_RNDN);
                mpfr_set_d(mp_rate, rate, MPFR_RNDN);
                mpfr_set_si(mp_years, years, MPFR_RNDN);

                if (rate == 0.0 || years == 0) {{
                    mpfr_set(mp_result, mp_principal, MPFR_RNDN);
                }} else {{
                    mpfr_add_d(mp_base, mp_rate, 1.0, MPFR_RNDN);
                    mpfr_pow(mp_result, mp_base, mp_years, MPFR_RNDN);
                    mpfr_mul(mp_result, mp_result, mp_principal, MPFR_RNDN);
                }}

                double mpfr_val = mpfr_get_d(mp_result, MPFR_RNDN);
                double rel_error = 0.0;
                if (mpfr_val != 0.0) {{
                    rel_error = fabs((user_val - mpfr_val) / mpfr_val);
                }} else {{
                    rel_error = fabs(user_val - mpfr_val);
                }}

                printf("%.2f, %.9f, %d, %.15e, %.15e, %.15e\\n",
                       principal, rate, years, user_val, mpfr_val, rel_error);

                mpfr_clear(mp_principal);
                mpfr_clear(mp_rate);
                mpfr_clear(mp_years);
                mpfr_clear(mp_result);
                mpfr_clear(mp_base);
            }}
        }}
    }}

    return 0;
}}'''
        return test_code

    def generate_generic_test(self, user_code: Optional[str], target: str, samples: List[float], func_type: str) -> Optional[str]:
        """生成通用测试代码（对于未知函数类型）"""
        if not user_code and not target:
            return None

        # 简单的数学函数测试模板
        func_impl = user_code if user_code else f"// {target}"

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    printf("# Generic test for {func_type}\\n");
    printf("# User code or target: {target[:100]}...\\n");
    return 0;
}}'''
        return test_code

    def compile_and_run_test(self, case_id: int, test_code: str) -> Optional[List[Tuple[float, float, float]]]:
        """编译并运行测试代码"""
        if not test_code:
            return None

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(test_code)
            c_file = f.name

        try:
            # 编译
            exe_file = c_file.replace('.c', '')

            # 尝试不同的编译选项
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

            # 运行
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
            # 清理临时文件
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

            # 提取信息
            input_content = case_data['input'][0]['content']
            output_content = case_data['output']['content']

            input_range = self.extract_input_range(input_content)
            samples = self.extract_test_samples(input_content)
            func_type = self.determine_function_type(input_content)

            print(f"  函数类型: {func_type}")
            print(f"  输入范围: {input_range}")
            print(f"  测试样本: {samples[:5]}...")

            # 生成测试代码
            test_code = self.generate_test_code(i, case_data, samples, func_type)
            if not test_code:
                print(f"  无法生成测试代码")
                continue

            # 编译运行
            errors = self.compile_and_run_test(i, test_code)
            if errors:
                print(f"  获得 {len(errors)} 个误差数据点")

                # 保存案例误差统计
                self.save_case_errors(i, errors, func_type, input_range)

                # 添加到全局误差列表
                for x_val, user_val, rel_error in errors:
                    self.all_errors.append((i, x_val, user_val, rel_error))
            else:
                print(f"  未获得有效误差数据")

    def save_case_errors(self, case_id: int, errors: List[Tuple[float, float, float]],
                        func_type: str, input_range: str):
        """保存单个案例的误差统计"""
        sorted_errors = sorted(errors, key=lambda x: x[2], reverse=True)[:5]

        output_file = os.path.join(self.output_dir, f"case{case_id+1}_error_stats.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 案例 {case_id+1} 误差统计\n\n")
            f.write(f"**函数类型**: {func_type}\n")
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
        """生成汇总报告"""
        if not self.all_errors:
            print("没有误差数据，无法生成汇总报告")
            return

        # 按误差排序，取最大的10个
        sorted_errors = sorted(self.all_errors, key=lambda x: x[3], reverse=True)[:10]

        output_file = os.path.join(self.output_dir, "summary_report.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 浮点误差分析汇总报告\n\n")
            f.write(f"**分析时间**: {self.mpfr_precision} 位精度\n")
            f.write(f"**总案例数**: {len(self.cases)}\n")
            f.write(f"**有效误差数据点**: {len(self.all_errors)}\n\n")

            f.write("## 误差最大的10个样本（所有案例）\n\n")
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
    # 检查环境
    try:
        result = subprocess.run(['gcc', '--version'], capture_output=True)
        if result.returncode != 0:
            print("错误: 未找到gcc编译器")
            sys.exit(1)
    except:
        print("错误: 未找到gcc编译器")
        sys.exit(1)

    # 创建分析器实例
    analyzer = FixedMPFRAnalyzer(
        input_file="模型代码生成浮点数值计算能力评测.json",
        output_dir="./results/",
        mpfr_precision=128
    )

    print("=== 修复版MPFR浮点误差分析工具 ===")
    print(f"输入文件: {analyzer.input_file}")
    print(f"输出目录: {analyzer.output_dir}")
    print(f"MPFR精度: {analyzer.mpfr_precision} 位")

    # 执行分析
    analyzer.load_cases()
    analyzer.process_all_cases()
    analyzer.generate_summary_report()

    print(f"\n分析完成！结果保存在 {analyzer.output_dir}")


if __name__ == "__main__":
    main()
