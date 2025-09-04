#!/usr/bin/env python3
"""
自动化浮点误差分析工具
使用MPFR作为Oracle，统计相对浮点误差
"""

import json
import re
import os
import subprocess
import sys
from typing import List, Dict, Tuple, Optional
import tempfile
import shutil

class MPFRErrorAnalyzer:
    def __init__(self, input_file: str, output_dir: str, mpfr_precision: int = 128):
        self.input_file = input_file
        self.output_dir = output_dir
        self.mpfr_precision = mpfr_precision
        self.cases = []
        self.all_errors = []

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def load_cases(self):
        """加载JSON案例文件"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 分割多个JSON对象
            json_objects = []
            current_obj = ""
            brace_count = 0

            for line in content.split('\n'):
                if line.strip():
                    current_obj += line + '\n'
                    brace_count += line.count('{') - line.count('}')

                    if brace_count == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            json_objects.append(obj)
                            current_obj = ""
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
            r'rate∈\[(.*?)\]'
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        return None

    def extract_test_samples(self, content: str) -> List[float]:
        """提取测试样本"""
        samples = []

        # 匹配各种形式的测试样本
        patterns = [
            r'x=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'\(x=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\)',
            r'样本.*?x=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
            r'测试.*?x=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    value = float(match)
                    if value not in samples:
                        samples.append(value)
                except ValueError:
                    continue

        # 如果没有找到样本，添加一些默认值
        if not samples:
            samples = [0.0, 1e-8, -1e-8, 5e-9, -3e-9]

        # 特殊处理零值样本
        if 0.0 not in samples:
            samples.append(0.0)
        if -0.0 not in samples:
            samples.append(-0.0)

        return samples[:10]  # 最多取10个样本

    def extract_c_code(self, content: str) -> Optional[str]:
        """提取C代码"""
        # 匹配 ```c ... ``` 代码块
        pattern = r'```c\s*\n(.*?)\n```'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1)

        # 尝试其他模式
        pattern = r'```\s*\n(.*?)\n```'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            code = match.group(1)
            if 'double' in code and 'return' in code:
                return code

        return None

    def determine_function_type(self, content: str) -> str:
        """确定函数类型和对应的MPFR函数"""
        content_lower = content.lower()

        if 'exp_minus_one' in content_lower or 'expm1' in content_lower:
            return 'expm1'
        elif 'optimized_sin' in content_lower or 'sin(' in content_lower:
            return 'sin'
        elif 'compound_value' in content_lower or '复利' in content:
            return 'compound'
        elif 'implied_volatility' in content_lower or '波动率' in content:
            return 'volatility'
        elif 'internal_rate_of_return' in content_lower or 'irr' in content_lower:
            return 'irr'
        else:
            return 'unknown'

    def generate_test_code(self, case_id: int, case_data: Dict, samples: List[float], func_type: str) -> str:
        """生成测试C代码"""
        user_code = self.extract_c_code(case_data['output']['content'])
        if not user_code:
            return None

        # 根据函数类型选择合适的测试代码模板
        if func_type == 'expm1':
            return self.generate_expm1_test(user_code, samples)
        elif func_type == 'sin':
            return self.generate_sin_test(user_code, samples)
        elif func_type == 'compound':
            return self.generate_compound_test(user_code, samples)
        else:
            return self.generate_generic_test(user_code, samples, func_type)

    def generate_expm1_test(self, user_code: str, samples: List[float]) -> str:
        """生成expm1函数测试代码"""
        samples_str = ', '.join([f"{x:.15e}" for x in samples])

        return f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{user_code}

int main() {{
    mpfr_t x, ref, user_result;
    mpfr_init2(x, {self.mpfr_precision});
    mpfr_init2(ref, {self.mpfr_precision});
    mpfr_init2(user_result, {self.mpfr_precision});

    double test_samples[] = {{{samples_str}}};
    int n_samples = sizeof(test_samples) / sizeof(test_samples[0]);

    printf("# x_value, user_value, mpfr_value, relative_error\\n");

    for (int i = 0; i < n_samples; i++) {{
        double x_val = test_samples[i];

        // 设置MPFR变量
        mpfr_set_d(x, x_val, MPFR_RNDN);

        // 计算MPFR参考值
        mpfr_expm1(ref, x, MPFR_RNDN);

        // 计算用户函数值
        double user_val = exp_minus_one(x_val);
        mpfr_set_d(user_result, user_val, MPFR_RNDN);

        // 计算相对误差
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

    def generate_sin_test(self, user_code: str, samples: List[float]) -> str:
        """生成sin函数测试代码"""
        samples_str = ', '.join([f"{x:.15e}" for x in samples])

        return f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{user_code}

int main() {{
    mpfr_t x, ref, user_result;
    mpfr_init2(x, {self.mpfr_precision});
    mpfr_init2(ref, {self.mpfr_precision});
    mpfr_init2(user_result, {self.mpfr_precision});

    double test_samples[] = {{{samples_str}}};
    int n_samples = sizeof(test_samples) / sizeof(test_samples[0]);

    printf("# x_value, user_value, mpfr_value, relative_error\\n");

    for (int i = 0; i < n_samples; i++) {{
        double x_val = test_samples[i];

        // 设置MPFR变量
        mpfr_set_d(x, x_val, MPFR_RNDN);

        // 计算MPFR参考值
        mpfr_sin(ref, x, MPFR_RNDN);

        // 计算用户函数值
        double user_val = optimized_sin(x_val);
        mpfr_set_d(user_result, user_val, MPFR_RNDN);

        // 计算相对误差
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

    def generate_compound_test(self, user_code: str, samples: List[float]) -> str:
        """生成复利计算测试代码"""
        return f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{user_code}

int main() {{
    // 复利计算测试
    double principal = 1000.0;
    double rates[] = {{0.05, 0.001, 1e-6, 0.0}};
    int years[] = {{10, 50, 100}};

    printf("# principal, rate, years, user_value, mpfr_value, relative_error\\n");

    for (int i = 0; i < 4; i++) {{
        for (int j = 0; j < 3; j++) {{
            double rate = rates[i];
            int year = years[j];

            // 用户函数值
            double user_val = compound_value(principal, rate, year);

            // MPFR参考值
            mpfr_t mp_principal, mp_rate, mp_year, mp_result;
            mpfr_init2(mp_principal, {self.mpfr_precision});
            mpfr_init2(mp_rate, {self.mpfr_precision});
            mpfr_init2(mp_year, {self.mpfr_precision});
            mpfr_init2(mp_result, {self.mpfr_precision});

            mpfr_set_d(mp_principal, principal, MPFR_RNDN);
            mpfr_set_d(mp_rate, rate, MPFR_RNDN);
            mpfr_set_si(mp_year, year, MPFR_RNDN);

            if (rate == 0.0) {{
                mpfr_set(mp_result, mp_principal, MPFR_RNDN);
            }} else {{
                mpfr_add_d(mp_result, mp_rate, 1.0, MPFR_RNDN);
                mpfr_pow(mp_result, mp_result, mp_year, MPFR_RNDN);
                mpfr_mul(mp_result, mp_result, mp_principal, MPFR_RNDN);
            }}

            double mpfr_val = mpfr_get_d(mp_result, MPFR_RNDN);

            // 计算相对误差
            double rel_error = 0.0;
            if (mpfr_val != 0.0) {{
                rel_error = fabs((user_val - mpfr_val) / mpfr_val);
            }} else {{
                rel_error = fabs(user_val - mpfr_val);
            }}

            printf("%.6f, %.9f, %d, %.15e, %.15e, %.15e\\n",
                   principal, rate, year, user_val, mpfr_val, rel_error);

            mpfr_clear(mp_principal);
            mpfr_clear(mp_rate);
            mpfr_clear(mp_year);
            mpfr_clear(mp_result);
        }}
    }}

    return 0;
}}'''

    def generate_generic_test(self, user_code: str, samples: List[float], func_type: str) -> str:
        """生成通用测试代码"""
        samples_str = ', '.join([f"{x:.15e}" for x in samples])

        return f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{user_code}

int main() {{
    printf("# 无法生成针对性测试，函数类型: {func_type}\\n");
    printf("# 用户代码已包含但未能识别函数类型\\n");
    return 1;
}}'''

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
            compile_cmd = ['gcc', '-o', exe_file, c_file, '-lmpfr', '-lgmp', '-lm']

            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"案例 {case_id} 编译失败:")
                print(result.stderr)
                return None

            # 运行
            run_result = subprocess.run([exe_file], capture_output=True, text=True)
            if run_result.returncode != 0:
                print(f"案例 {case_id} 运行失败:")
                print(run_result.stderr)
                return None

            # 解析结果
            errors = []
            for line in run_result.stdout.strip().split('\\n'):
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
            print(f"案例 {case_id} 处理异常: {e}")
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
            print(f"\\n处理案例 {i+1}/{len(self.cases)}...")

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
        # 按误差排序，取最大的5个
        sorted_errors = sorted(errors, key=lambda x: x[2], reverse=True)[:5]

        output_file = os.path.join(self.output_dir, f"case{case_id+1}_error_stats.md")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 案例 {case_id+1} 误差统计\\n\\n")
            f.write(f"**函数类型**: {func_type}\\n")
            f.write(f"**输入范围**: {input_range}\\n")
            f.write(f"**MPFR精度**: {self.mpfr_precision} 位\\n\\n")

            f.write("## 误差最大的5个样本\\n\\n")
            f.write("| x值 | 用户函数值 | 相对误差 |\\n")
            f.write("|-----|-----------|----------|\\n")

            for x_val, user_val, rel_error in sorted_errors:
                f.write(f"| {x_val:.6e} | {user_val:.6e} | {rel_error:.6e} |\\n")

            f.write(f"\\n**总测试样本数**: {len(errors)}\\n")
            if errors:
                avg_error = sum(e[2] for e in errors) / len(errors)
                f.write(f"**平均相对误差**: {avg_error:.6e}\\n")

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.all_errors:
            print("没有误差数据，无法生成汇总报告")
            return

        # 按误差排序，取最大的5个
        sorted_errors = sorted(self.all_errors, key=lambda x: x[3], reverse=True)[:5]

        summary_file = os.path.join(self.output_dir, "summary.md")

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# 浮点误差分析汇总报告\\n\\n")
            f.write(f"**分析案例数**: {len(self.cases)}\\n")
            f.write(f"**MPFR精度**: {self.mpfr_precision} 位\\n")
            f.write(f"**总误差数据点**: {len(self.all_errors)}\\n\\n")

            f.write("## 误差最大的5个样本（所有案例）\\n\\n")
            f.write("| 案例ID | x值 | 用户函数值 | 相对误差 |\\n")
            f.write("|--------|-----|-----------|----------|\\n")

            for case_id, x_val, user_val, rel_error in sorted_errors:
                f.write(f"| case{case_id+1} | {x_val:.6e} | {user_val:.6e} | {rel_error:.6e} |\\n")

            # 统计信息
            if self.all_errors:
                avg_error = sum(e[3] for e in self.all_errors) / len(self.all_errors)
                max_error = max(e[3] for e in self.all_errors)
                min_error = min(e[3] for e in self.all_errors)

                f.write(f"\\n## 误差统计\\n\\n")
                f.write(f"- **最大相对误差**: {max_error:.6e}\\n")
                f.write(f"- **最小相对误差**: {min_error:.6e}\\n")
                f.write(f"- **平均相对误差**: {avg_error:.6e}\\n")

def main():
    # 检查MPFR是否可用
    try:
        result = subprocess.run(['gcc', '--version'], capture_output=True)
        if result.returncode != 0:
            print("错误: 未找到gcc编译器")
            sys.exit(1)
    except:
        print("错误: 未找到gcc编译器")
        sys.exit(1)

    try:
        result = subprocess.run(['pkg-config', '--exists', 'mpfr'], capture_output=True)
        if result.returncode != 0:
            print("警告: 未找到MPFR库，尝试直接链接...")
    except:
        print("警告: 未找到pkg-config或MPFR库")

    # 创建分析器实例
    analyzer = MPFRErrorAnalyzer(
        input_file="代码生成浮点数值计算能力评测.json",
        output_dir="./results/",
        mpfr_precision=128
    )

    print("=== MPFR浮点误差分析工具 ===")
    print(f"输入文件: {analyzer.input_file}")
    print(f"输出目录: {analyzer.output_dir}")
    print(f"MPFR精度: {analyzer.mpfr_precision} 位")

    # 执行分析
    analyzer.load_cases()
    analyzer.process_all_cases()
    analyzer.generate_summary_report()

    print(f"\\n分析完成！结果保存在 {analyzer.output_dir}")

if __name__ == "__main__":
    main()
