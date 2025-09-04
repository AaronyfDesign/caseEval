#!/usr/bin/env python3
"""
增强版浮点数值计算代码生成器
针对模型代码生成浮点数值计算能力评测.json文件中的具体case进行优化
添加更多符合case要求的代码生成模板
"""

import json
import re
import os
import subprocess
import sys
from typing import List, Dict, Tuple, Optional
import tempfile

class EnhancedFloatingPointCodeGenerator:
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
                            except json.JSONDecodeError as e:
                                pass

            self.cases = json_objects
            print(f"成功加载 {len(self.cases)} 个案例")

        except Exception as e:
            print(f"加载案例文件失败: {e}")
            sys.exit(1)

    def extract_function_info(self, content: str) -> Dict[str, str]:
        """提取函数信息"""
        info = {}

        # 提取函数名
        func_patterns = [
            r'function\s+double\s+(\w+)\(',
            r'Implement\s+function\s+double\s+(\w+)\(',
            r'struct\s+(\w+)\s+(\w+)\(',
        ]

        for pattern in func_patterns:
            match = re.search(pattern, content)
            if match:
                info['function_name'] = match.group(1)
                break

        # 提取输入范围
        range_patterns = [
            r'x∈\[(.*?)\]',
            r'input\s+range[:\s]+.*?∈\[(.*?)\]',
            r'ranges?[:\s].*?∈\[(.*?)\]',
        ]

        for pattern in range_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                info['input_range'] = match.group(1)
                break

        # 提取特殊要求
        requirements = []
        if 'catastrophic cancellation' in content.lower():
            requirements.append('avoid_cancellation')
        if 'taylor' in content.lower() or 'expansion' in content.lower():
            requirements.append('taylor_expansion')
        if 'precision' in content.lower():
            requirements.append('high_precision')
        if 'overflow' in content.lower():
            requirements.append('overflow_protection')
        if 'underflow' in content.lower():
            requirements.append('underflow_protection')
        if 'ieee 754' in content.lower():
            requirements.append('ieee754_compliant')
        if 'negative zero' in content.lower():
            requirements.append('preserve_negative_zero')
        if 'subnormal' in content.lower():
            requirements.append('handle_subnormal')
        if 'nan' in content.lower() or 'inf' in content.lower():
            requirements.append('handle_special_values')
        if 'newton' in content.lower() and 'raphson' in content.lower():
            requirements.append('newton_raphson')
        if 'bisection' in content.lower():
            requirements.append('bisection_method')
        if 'kahan' in content.lower():
            requirements.append('kahan_summation')
        if 'mixed precision' in content.lower() or 'fp128' in content.lower():
            requirements.append('mixed_precision')

        info['requirements'] = requirements
        return info

    def determine_function_category(self, content: str, target: str) -> str:
        """确定函数类别"""
        content_lower = content.lower()
        target_lower = target.lower()

        categories = {
            'transcendental': ['exp_minus_one', 'expm1', 'sin', 'cos', 'sinh', 'tanh', 'log1p', 'sqrt'],
            'financial': ['compound_value', 'monthly_payment', 'bond_duration', 'implied_volatility',
                         'internal_rate_of_return', 'value_at_risk'],
            'optimization': ['global_min', 'lyapunov_exponent'],
            'integration': ['adaptive_integrate', 'adaptive_singular_integral'],
            'linear_algebra': ['extreme_eigenpair'],
            'series': ['direct_sum', 'levin_acceleration', 'borel_sum'],
            'distance': ['haversine_distance'],
        }

        for category, functions in categories.items():
            for func in functions:
                if func in content_lower or func in target_lower:
                    return category

        return 'general'

    def generate_exp_minus_one_template(self, info: Dict) -> str:
        """生成exp(x)-1函数的优化模板"""
        return """
double exp_minus_one(double x) {
    // 对于极小的 |x|，使用泰勒级数避免相消误差
    // exp(x) - 1 = x + x²/2! + x³/3! + x⁴/4! + ...

    if (x == 0.0) {
        return 0.0;
    }

    // 对于 |x| < 1e-5，使用泰勒展开
    if (fabs(x) < 1e-5) {
        double x2 = x * x;
        double x3 = x2 * x;
        double x4 = x3 * x;
        double x5 = x4 * x;

        // 使用Horner方法计算泰勒级数
        // x * (1 + x/2 * (1 + x/3 * (1 + x/4 * (1 + x/5))))
        return x * (1.0 + x * (0.5 + x * (1.0/6.0 + x * (1.0/24.0 + x * (1.0/120.0)))));
    }

    // 对于较大的 |x|，使用标准库函数
    return expm1(x);
}
"""

    def generate_optimized_sin_template(self, info: Dict) -> str:
        """生成优化sin函数的模板"""
        return """
double optimized_sin(double x) {
    // 处理特殊值
    if (isnan(x)) return x;
    if (isinf(x)) return NAN;
    if (x == 0.0) return x;  // 保持负零

    // 对于 |x| < 1e-5，使用泰勒展开优化精度
    if (fabs(x) < 1e-5) {
        double x2 = x * x;
        // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
        return x * (1.0 - x2/6.0 * (1.0 - x2/20.0 * (1.0 - x2/42.0)));
    }

    return sin(x);
}
"""

    def generate_compound_value_template(self, info: Dict) -> str:
        """生成复利计算函数模板"""
        return """
double compound_value(double principal, double rate, int years) {
    // 零利率或零年数情况
    if (rate == 0.0 || years == 0) {
        return principal;
    }

    // 小利率高精度处理
    if (rate < 1e-5) {
        // 使用 exp(years * log1p(rate)) 避免精度损失
        double log_term = log1p(rate);
        return principal * exp(years * log_term);
    }

    // 长期计算优化
    if (years > 50) {
        double log_factor = log1p(rate);
        double exponent = years * log_factor;

        // 溢出检查
        if (exponent > 700) {
            return INFINITY;
        }

        return principal * exp(exponent);
    }

    // 标准情况
    return principal * pow(1.0 + rate, years);
}
"""

    def generate_sqrt_optimized_template(self, info: Dict) -> str:
        """生成优化平方根函数模板"""
        return """
double sqrt_optimized(double x) {
    // IEEE 754兼容性检查
    if (x == 0.0) return x;  // 保持-0.0
    if (x < 0.0) return NAN;
    if (isnan(x)) return x;
    if (isinf(x)) return x;

    // 次正规数优化
    if (x < 1e-300) {
        const double scale_factor = 0x1p+512;  // 2^512
        const double unscale_factor = 0x1p-256; // 2^-256
        double scaled_x = x * scale_factor;
        double result = sqrt(scaled_x);
        return result * unscale_factor;
    }

    // 大数溢出防护
    if (x > 1e150) {
        const double scale_factor = 0x1p-512;
        const double unscale_factor = 0x1p+256;
        double scaled_x = x * scale_factor;
        double result = sqrt(scaled_x);
        return result * unscale_factor;
    }

    return sqrt(x);
}
"""

    def generate_log1p_optimized_template(self, info: Dict) -> str:
        """生成优化log(1+x)函数模板"""
        return """
double log1p_optimized(double x) {
    if (x == -1.0) return -INFINITY;
    if (isnan(x)) return x;
    if (isinf(x)) return (x > 0) ? x : NAN;

    // 小值高精度处理
    if (fabs(x) < 1e-5) {
        // log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
        return x * (1.0 - x * (0.5 - x * (1.0/3.0 - x * 0.25)));
    }

    // 大值渐近展开
    if (x > 1e15) {
        return log(x) + 1.0/x - 0.5/(x*x);
    }

    return log(1.0 + x);
}
"""

    def generate_sinh_optimized_template(self, info: Dict) -> str:
        """生成优化双曲正弦函数模板"""
        return """
double sinh_optimized(double x) {
    if (x == 0.0) return x;  // 保持符号

    double abs_x = fabs(x);

    // 极小值泰勒展开
    if (abs_x < 1e-8) {
        double x2 = x * x;
        // sinh(x) ≈ x + x³/6 + x⁵/120
        return x * (1.0 + x2 * (1.0/6.0 + x2 * (1.0/120.0)));
    }

    // 大值防溢出
    if (abs_x > 50.0) {
        double sign = (x > 0) ? 1.0 : -1.0;
        return sign * 0.5 * exp(abs_x);
    }

    return 0.5 * (exp(x) - exp(-x));
}
"""

    def generate_tanh_optimized_template(self, info: Dict) -> str:
        """生成优化双曲正切函数模板"""
        return """
double tanh_optimized(double x) {
    if (x == 0.0) return x;

    // 饱和区域
    if (x > 20.0) return 1.0;
    if (x < -20.0) return -1.0;

    // 极小值处理
    if (fabs(x) < 1e-8) {
        double x2 = x * x;
        // tanh(x) ≈ x - x³/3 + 2x⁵/15
        return x * (1.0 - x2 * (1.0/3.0 - x2 * (2.0/15.0)));
    }

    // 防exp(2x)溢出
    if (fabs(x) > 50.0) {
        return (x > 0) ? 1.0 : -1.0;
    }

    double ex = exp(x);
    double e_x = exp(-x);
    return (ex - e_x) / (ex + e_x);
}
"""

    def generate_financial_template(self, func_name: str, info: Dict) -> str:
        """生成金融函数模板"""
        if func_name == 'monthly_payment':
            return """
double monthly_payment(double principal, double rate, int term) {
    // 输入验证
    if (principal < 1000.0 || principal > 1e9) return NAN;
    if (rate < 0.0 || rate > 0.5) return NAN;
    if (term < 1 || term > 360) return NAN;

    // 零利率特殊情况
    if (fabs(rate) < 1e-10) {
        return principal / term;
    }

    double monthly_rate = rate / 12.0;

    // 单期贷款
    if (term == 1) {
        return principal * (1.0 + monthly_rate);
    }

    // 极小利率优化
    if (monthly_rate < 0.001) {
        double n = term;
        double ln_discount = -n * log1p(monthly_rate);
        double discount_factor = exp(ln_discount);
        double denominator = 1.0 - discount_factor;

        if (fabs(denominator) < 1e-15) {
            return principal / term * (1.0 + monthly_rate * (n + 1.0) / 2.0);
        }

        return principal * monthly_rate / denominator;
    }

    // 标准年金公式
    double factor = pow(1.0 + monthly_rate, term);
    return principal * monthly_rate * factor / (factor - 1.0);
}
"""
        elif func_name == 'bond_duration':
            return """
double bond_duration(double coupon_rate, double yield, int periods, double face_value) {
    if (periods <= 0 || face_value <= 0.0) return 0.0;

    // 零收益率简化公式
    if (yield == 0.0) {
        double coupon = coupon_rate * face_value;
        double total_weighted_time = 0.0;
        double total_pv = 0.0;

        for (int t = 1; t <= periods; ++t) {
            double cf = (t == periods) ? (coupon + face_value) : coupon;
            total_pv += cf;
            total_weighted_time += t * cf;
        }

        return total_weighted_time / total_pv;
    }

    double coupon = coupon_rate * face_value;
    double price = 0.0;
    double duration_weighted = 0.0;

    for (int t = 1; t <= periods; ++t) {
        double discount_factor;

        // 极小收益率优化
        if (yield < 0.001) {
            discount_factor = exp(-t * yield * (1.0 - yield * (t-1) / 2.0));
        } else {
            discount_factor = pow(1.0 + yield, -t);
        }

        double cf = (t == periods) ? (coupon + face_value) : coupon;
        double cf_pv = cf * discount_factor;

        price += cf_pv;
        duration_weighted += t * cf_pv;
    }

    return duration_weighted / price;
}
"""
        return ""

    def generate_optimization_template(self, func_name: str, info: Dict) -> str:
        """生成优化算法模板"""
        if func_name == 'global_min':
            return """
Point global_min(double (*f)(double, double), Point domain) {
    Point result = {0, 0, INFINITY, false, 0};
    long long evals = 0;
    const long long MAX_EVALS = 1000000;

    // 多起点全局搜索
    const int n_starts = 50;
    std::vector<Point> candidates;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // 拉丁超立方采样
    for (int i = 0; i < n_starts && evals < MAX_EVALS; ++i) {
        Point start = {dis(gen), dis(gen), 0, false, 0};
        start.f = f(start.x, start.y);
        evals++;

        // 局部优化（梯度下降）
        Point local_min = local_optimize(f, start, evals, MAX_EVALS);
        candidates.push_back(local_min);

        if (local_min.f < result.f) {
            result = local_min;
        }
    }

    // 高精度验证最佳候选解
    result = verify_with_fp128(f, result);
    result.evals = evals;

    return result;
}
"""
        return ""

    def generate_integration_template(self, func_name: str, info: Dict) -> str:
        """生成积分算法模板"""
        if func_name == 'adaptive_integrate':
            return """
double adaptive_integrate(double (*f)(double), double a, double b) {
    const int MAX_EVALS = 10000000;
    const double TARGET_TOL = 1e-20;
    int evals = 0;

    // 无穷区间变换
    if (b == INFINITY) {
        // 使用变换 x = t/(1-t), dx = dt/(1-t)²
        auto g = [f](double t) -> double {
            if (t >= 1.0) return 0.0;
            double x = t / (1.0 - t);
            double jacobian = 1.0 / ((1.0 - t) * (1.0 - t));
            return f(x) * jacobian;
        };

        return adaptive_simpson(g, 0.0, 0.999999, TARGET_TOL, evals, MAX_EVALS);
    }

    return adaptive_simpson(f, a, b, TARGET_TOL, evals, MAX_EVALS);
}

double adaptive_simpson(double (*f)(double), double a, double b, double tol, int& evals, int max_evals) {
    if (evals >= max_evals) return 0.0;

    double h = (b - a) / 2.0;
    double m = (a + b) / 2.0;

    double fa = f(a); evals++;
    double fm = f(m); evals++;
    double fb = f(b); evals++;

    double S1 = h/3.0 * (fa + 4*fm + fb);

    double m1 = (a + m) / 2.0;
    double m2 = (m + b) / 2.0;
    double fm1 = f(m1); evals++;
    double fm2 = f(m2); evals++;

    double S2 = h/6.0 * (fa + 4*fm1 + 2*fm + 4*fm2 + fb);
    double error = fabs(S2 - S1) / 15.0;

    if (error < tol || evals >= max_evals) {
        return S2;
    }

    return adaptive_simpson(f, a, m, tol/2, evals, max_evals) +
           adaptive_simpson(f, m, b, tol/2, evals, max_evals);
}
"""
        return ""

    def generate_test_code(self, case_id: int, case_data: Dict, func_category: str, func_info: Dict) -> str:
        """生成测试代码"""
        func_name = func_info.get('function_name', 'unknown')
        target = case_data.get('target', '')

        # 根据类别选择模板
        if func_category == 'transcendental':
            if 'exp_minus_one' in func_name or 'expm1' in target:
                template = self.generate_exp_minus_one_template(func_info)
            elif 'sin' in func_name:
                template = self.generate_optimized_sin_template(func_info)
            elif 'sqrt' in func_name:
                template = self.generate_sqrt_optimized_template(func_info)
            elif 'log1p' in func_name:
                template = self.generate_log1p_optimized_template(func_info)
            elif 'sinh' in func_name:
                template = self.generate_sinh_optimized_template(func_info)
            elif 'tanh' in func_name:
                template = self.generate_tanh_optimized_template(func_info)
            else:
                template = "// Generic transcendental function template"

        elif func_category == 'financial':
            template = self.generate_financial_template(func_name, func_info)

        elif func_category == 'optimization':
            template = self.generate_optimization_template(func_name, func_info)

        elif func_category == 'integration':
            template = self.generate_integration_template(func_name, func_info)

        else:
            template = "// Generic function template"

        # 生成完整的测试代码
        test_code = f"""
#include <mpfr.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

{template}

int main() {{
    // 测试案例 {case_id + 1}: {func_name}
    printf("Testing {func_name}\\n");

    // 这里会添加具体的测试逻辑
    return 0;
}}
"""
        return test_code

    def process_all_cases(self):
        """处理所有案例，生成优化的代码模板"""
        for i, case_data in enumerate(self.cases):
            print(f"\n处理案例 {i+1}/{len(self.cases)}...")

            input_content = case_data['input'][0]['content']
            target = case_data.get('target', '')

            # 提取函数信息
            func_info = self.extract_function_info(input_content)
            func_category = self.determine_function_category(input_content, target)

            print(f"  函数类别: {func_category}")
            print(f"  函数名: {func_info.get('function_name', 'unknown')}")
            print(f"  特殊要求: {', '.join(func_info.get('requirements', []))}")

            # 生成优化的代码模板
            test_code = self.generate_test_code(i, case_data, func_category, func_info)

            # 保存到文件
            output_file = os.path.join(self.output_dir, f"case{i+1}_{func_category}_template.c")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(test_code)

            print(f"  已生成模板: {output_file}")

    def generate_summary_report(self):
        """生成汇总报告"""
        summary_file = os.path.join(self.output_dir, "code_generation_summary.md")

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# 浮点数值计算代码生成汇总报告\n\n")
            f.write(f"**总案例数**: {len(self.cases)}\n")
            f.write(f"**MPFR精度**: {self.mpfr_precision} 位\n\n")

            # 统计各类别函数
            categories = {}
            for i, case_data in enumerate(self.cases):
                input_content = case_data['input'][0]['content']
                target = case_data.get('target', '')
                func_category = self.determine_function_category(input_content, target)

                if func_category not in categories:
                    categories[func_category] = []
                categories[func_category].append(i + 1)

            f.write("## 函数类别分布\n\n")
            for category, cases in categories.items():
                f.write(f"- **{category}**: {len(cases)} 个案例 (case {', case '.join(map(str, cases))})\n")

            f.write("\n## 优化技术应用统计\n\n")

            techniques = {
                'taylor_expansion': '泰勒展开',
                'avoid_cancellation': '避免相消误差',
                'high_precision': '高精度计算',
                'overflow_protection': '溢出保护',
                'ieee754_compliant': 'IEEE 754兼容',
                'mixed_precision': '混合精度',
                'newton_raphson': 'Newton-Raphson方法',
                'kahan_summation': 'Kahan求和'
            }

            technique_count = {tech: 0 for tech in techniques}

            for case_data in self.cases:
                input_content = case_data['input'][0]['content']
                func_info = self.extract_function_info(input_content)

                for req in func_info.get('requirements', []):
                    if req in technique_count:
                        technique_count[req] += 1

            for tech, count in technique_count.items():
                if count > 0:
                    f.write(f"- **{techniques[tech]}**: {count} 个案例\n")

def main():
    generator = EnhancedFloatingPointCodeGenerator(
        input_file="模型代码生成浮点数值计算能力评测.json",
        output_dir="./enhanced_templates/",
        mpfr_precision=128
    )

    print("=== 增强版浮点数值计算代码生成器 ===")
    print(f"输入文件: {generator.input_file}")
    print(f"输出目录: {generator.output_dir}")

    generator.load_cases()
    generator.process_all_cases()
    generator.generate_summary_report()

    print(f"\n代码生成完成！模板保存在 {generator.output_dir}")

if __name__ == "__main__":
    main()
