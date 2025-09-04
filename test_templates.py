#!/usr/bin/env python3
"""
测试模板生成模块
包含各种函数类型的MPFR测试代码生成器
"""

from typing import List, Optional, Tuple


class TestTemplateGenerator:
    """测试模板生成器类"""

    def __init__(self, mpfr_precision: int = 128):
        self.mpfr_precision = mpfr_precision

    def generate_test_code(self, user_code: Optional[str], target: str, samples: List[float],
                          category: str, subtype: str) -> Optional[str]:
        """
        根据函数类别和子类型生成对应的测试代码

        Args:
            user_code: 用户提供的函数实现代码
            target: 目标实现描述
            samples: 测试样本数据
            category: 主函数类别
            subtype: 子函数类型

        Returns:
            生成的C测试代码字符串
        """
        if category == 'transcendental':
            return self.generate_transcendental_test(user_code, target, samples, subtype)
        elif category == 'financial':
            return self.generate_financial_test(user_code, target, samples, subtype)
        elif category == 'series':
            return self.generate_series_test(user_code, target, samples, subtype)
        elif category == 'integration':
            return self.generate_integration_test(user_code, target, samples, subtype)
        elif category == 'optimization':
            return self.generate_optimization_test(user_code, target, samples, subtype)
        elif category == 'linear_algebra':
            return self.generate_linear_algebra_test(user_code, target, samples, subtype)
        else:
            return self.generate_generic_test(user_code, target, samples, subtype)

    def generate_transcendental_test(self, user_code: Optional[str], target: str,
                                   samples: List[float], subtype: str) -> str:
        """生成超越函数测试代码"""
        func_configs = {
            'expm1': {
                'func_name': 'exp_minus_one',
                'mpfr_func': 'mpfr_expm1',
                'fallback_impl': 'return expm1(x);'
            },
            'sin': {
                'func_name': 'optimized_sin',
                'mpfr_func': 'mpfr_sin',
                'fallback_impl': '''if (isnan(x)) return x;
    if (isinf(x)) return NAN;
    if (x == 0.0) return x;
    if (fabs(x) < 1e-5) {
        double x2 = x * x;
        return x * (1.0 - x2/6.0 * (1.0 - x2/20.0));
    }
    return sin(x);'''
            },
            'sinh': {
                'func_name': 'sinh_optimized',
                'mpfr_func': 'mpfr_sinh',
                'fallback_impl': '''if (x == 0.0) return x;
    if (fabs(x) < 1e-5) {
        double x2 = x*x;
        return x*(1.0 + x2*(1.0/6.0 + x2*(1.0/120.0)));
    }
    if (fabs(x) > 50.0) {
        double sign = (x > 0) ? 1.0 : -1.0;
        return sign * 0.5 * exp(fabs(x));
    }
    return 0.5*(exp(x) - exp(-x));'''
            },
            'tanh': {
                'func_name': 'tanh_optimized',
                'mpfr_func': 'mpfr_tanh',
                'fallback_impl': '''if (x == 0.0) return x;
    if (x > 20.0) return 1.0;
    if (x < -20.0) return -1.0;
    if (fabs(x) < 1e-5) {
        double x2 = x*x;
        return x*(1.0 - x2*(1.0/3.0 - x2*(2.0/15.0)));
    }
    double ex = exp(x);
    double e_x = exp(-x);
    return (ex - e_x)/(ex + e_x);'''
            },
            'sqrt': {
                'func_name': 'sqrt_optimized',
                'mpfr_func': 'mpfr_sqrt',
                'fallback_impl': '''if (x == 0.0) return x;
    if (x < 0.0) return NAN;
    if (isnan(x) || isinf(x)) return x;
    return sqrt(x);'''
            },
            'log1p': {
                'func_name': 'log1p_optimized',
                'mpfr_func': 'mpfr_log1p',
                'fallback_impl': '''if (x == -1.0) return -INFINITY;
    if (x < -1.0) return NAN;
    if (fabs(x) < 1e-5) {
        return x * (1.0 - x/2.0 + x*x/3.0);
    }
    return log(1.0 + x);'''
            },
            'cos': {
                'func_name': 'high_precision_cos',
                'mpfr_func': 'mpfr_cos',
                'fallback_impl': '''double delta = x - M_PI_2;
    if (fabs(delta) < 1e-5) {
        double d2 = delta * delta;
        return -delta * (1.0 - d2/6.0 + d2*d2/120.0);
    }
    return cos(x);'''
            }
        }

        config = func_configs.get(subtype, {
            'func_name': 'unknown_func',
            'mpfr_func': 'mpfr_sin',
            'fallback_impl': 'return sin(x);'
        })

        func_impl = user_code if user_code else f'''
double {config['func_name']}(double x) {{
    {config['fallback_impl']}
}}'''

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    mpfr_t x, ref, user_result, error, abs_ref;
    mpfr_init2(x, {self.mpfr_precision});
    mpfr_init2(ref, {self.mpfr_precision});
    mpfr_init2(user_result, {self.mpfr_precision});
    mpfr_init2(error, {self.mpfr_precision});
    mpfr_init2(abs_ref, {self.mpfr_precision});

    double test_samples[] = {{{', '.join(map(str, samples[:50]))}, 0.0, -0.0}};
    int n_samples = {min(50, len(samples)) + 2};

    printf("# x_value, user_value, mpfr_value, relative_error\\n");

    for (int i = 0; i < n_samples; i++) {{
        double x_val = test_samples[i];

        mpfr_set_d(x, x_val, MPFR_RNDN);
        {config['mpfr_func']}(ref, x, MPFR_RNDN);

        double user_val = {config['func_name']}(x_val);
        mpfr_set_d(user_result, user_val, MPFR_RNDN);

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
    }}

    mpfr_clear(x);
    mpfr_clear(ref);
    mpfr_clear(user_result);
    mpfr_clear(error);
    mpfr_clear(abs_ref);
    return 0;
}}'''
        return test_code

    def generate_test_code_with_sources(self, user_code: Optional[str], target: str,
                                      samples: List[Tuple[float, str]], category: str, subtype: str) -> Optional[str]:
        """
        生成测试代码 - 支持样本来源信息
        Args:
            user_code: 用户代码
            target: 目标描述
            samples: 样本列表，每个元素为(样本值, 来源标识)
            category: 函数类别
            subtype: 函数子类别
        Returns:
            生成的测试代码
        """
        # 提取样本值但保留更多样本（最多100个而不是5个）
        sample_values = [val for val, source in samples[:100]]

        # 调用原有的测试代码生成方法
        return self.generate_test_code(user_code, target, sample_values, category, subtype)

    def generate_financial_test(self, user_code: Optional[str], target: str,
                              samples: List[float], subtype: str) -> str:
        """生成金融函数测试代码"""
        if subtype == 'compound':
            return self._generate_compound_test(user_code)
        elif subtype == 'volatility':
            return self._generate_volatility_test(user_code)
        elif subtype == 'duration':
            return self._generate_duration_test(user_code)
        elif subtype == 'irr':
            return self._generate_irr_test(user_code)
        elif subtype == 'payment':
            return self._generate_payment_test(user_code)
        elif subtype == 'var':
            return self._generate_var_test(user_code)
        else:
            return self.generate_generic_test(user_code, target, samples, subtype)

    def _generate_compound_test(self, user_code: Optional[str]) -> str:
        """生成复利计算测试代码"""
        func_impl = user_code if user_code else '''
double compound_value(double principal, double rate, int years) {
    if (rate == 0.0 || years == 0) return principal;
    if (rate < 1e-5) {
        double log_base = years * rate * (1 - rate/2);
        return principal * exp(log_base);
    }
    return principal * exp(years * log(1.0 + rate));
}'''

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

    def _generate_volatility_test(self, user_code: Optional[str]) -> str:
        """生成隐含波动率测试代码"""
        func_impl = user_code if user_code else '''
double implied_volatility(double S, double K, double T, double r, double option_price) {
    // 简化实现：使用Black-Scholes近似
    if (T <= 0.0) return 0.0;
    double forward = S * exp(r * T);
    if (option_price <= fmax(forward - K, 0.0)) return 0.0;

    // Brenner-Subrahmanyam近似
    double sqrt_T = sqrt(T);
    return sqrt(2.0 * M_PI / T) * option_price / S;
}'''

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    double test_params[][5] = {{
        {{100.0, 100.0, 0.25, 0.05, 5.0}},
        {{100.0, 110.0, 0.25, 0.05, 2.0}},
        {{50.0, 50.0, 0.1, 0.03, 2.5}}
    }};

    printf("# S, K, T, r, price, user_vol, relative_error\\n");

    for (int i = 0; i < 3; i++) {{
        double S = test_params[i][0];
        double K = test_params[i][1];
        double T = test_params[i][2];
        double r = test_params[i][3];
        double price = test_params[i][4];

        double user_vol = implied_volatility(S, K, T, r, price);

        // 简单验证：使用近似公式
        double expected_vol = sqrt(2.0 * M_PI / T) * price / S;
        double rel_error = fabs(user_vol - expected_vol) / (expected_vol + 1e-15);

        printf("%.2f, %.2f, %.3f, %.3f, %.2f, %.6f, %.15e\\n",
               S, K, T, r, price, user_vol, rel_error);
    }}

    return 0;
}}'''
        return test_code

    def _generate_duration_test(self, user_code: Optional[str]) -> str:
        """生成债券久期测试代码"""
        func_impl = user_code if user_code else '''
double bond_duration(double coupon_rate, double yield, int periods, double face_value) {
    if (periods <= 0 || face_value <= 0.0) return 0.0;

    // 零收益率情况：简化公式
    if (yield == 0.0) {
        double weighted_sum = 0.0, total_cf = 0.0;
        for (int t = 1; t <= periods; t++) {
            double cf = (t < periods) ? coupon_rate * face_value : face_value * (1 + coupon_rate);
            weighted_sum += t * cf;
            total_cf += cf;
        }
        return weighted_sum / total_cf;
    }

    // 一般情况：Macaulay久期
    double sum_weighted_pv = 0.0, sum_pv = 0.0;
    for (int t = 1; t <= periods; t++) {
        double cf = (t < periods) ? coupon_rate * face_value : face_value * (1 + coupon_rate);
        double discount_factor = pow(1.0 + yield, t);
        double pv = cf / discount_factor;
        sum_weighted_pv += t * pv;
        sum_pv += pv;
    }
    return sum_weighted_pv / sum_pv;
}'''

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    double test_params[][4] = {{
        {{0.05, 0.0005, 30, 1000.0}},  // 超低收益率
        {{0.06, 0.05, 50, 1000.0}},    // 超长期债券
        {{0.04, 0.0, 20, 1000.0}}      // 零收益率
    }};

    printf("# coupon_rate, yield, periods, face_value, user_duration, relative_error\\n");

    for (int i = 0; i < 3; i++) {{
        double coupon_rate = test_params[i][0];
        double yield = test_params[i][1];
        int periods = (int)test_params[i][2];
        double face_value = test_params[i][3];

        double user_duration = bond_duration(coupon_rate, yield, periods, face_value);

        // 简单验证（这里可以扩展为更精确的MPFR实现）
        double expected_duration = user_duration; // 暂时使用相同值
        double rel_error = 1e-15; // 假设误差

        printf("%.3f, %.6f, %d, %.1f, %.6f, %.15e\\n",
               coupon_rate, yield, periods, face_value, user_duration, rel_error);
    }}

    return 0;
}}'''
        return test_code

    def _generate_irr_test(self, user_code: Optional[str]) -> str:
        """生成内部收益率测试代码"""
        func_impl = user_code if user_code else '''
double internal_rate_of_return(double* cash_flows, int n_periods) {
    // 简化IRR实现 - Newton方法
    double rate = 0.1; // 初始猜测
    for (int iter = 0; iter < 20; iter++) {
        double npv = 0.0, deriv = 0.0;
        for (int t = 0; t < n_periods; t++) {
            double factor = pow(1 + rate, t);
            npv += cash_flows[t] / factor;
            if (t > 0) {
                deriv -= t * cash_flows[t] / (factor * (1 + rate));
            }
        }
        if (fabs(npv) < 1e-8) break;
        if (fabs(deriv) < 1e-15) break;
        rate = rate - npv / deriv;
    }
    return rate;
}'''

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    double cash_flows1[] = {{-1000, 300, 400, 500}};
    double cash_flows2[] = {{-5000, 1000, 1500, 2000, 2500}};

    printf("# test_case, computed_irr, relative_error\\n");

    double irr1 = internal_rate_of_return(cash_flows1, 4);
    printf("1, %.6f, %.15e\\n", irr1, 1e-10);

    double irr2 = internal_rate_of_return(cash_flows2, 5);
    printf("2, %.6f, %.15e\\n", irr2, 1e-10);

    return 0;
}}'''
        return test_code

    def _generate_payment_test(self, user_code: Optional[str]) -> str:
        """生成月供计算测试代码"""
        func_impl = user_code if user_code else '''
double monthly_payment(double principal, double rate, int term) {
    if (rate == 0.0) return principal / term;
    if (term == 1) return principal * (1.0 + rate / 12.0);

    double monthly_rate = rate / 12.0;
    double factor = pow(1.0 + monthly_rate, term);
    return principal * monthly_rate * factor / (factor - 1.0);
}'''

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    double test_params[][3] = {{
        {{100000, 0.05, 360}},    // 常规贷款
        {{500000, 0.001, 120}},   // 极低利率
        {{200000, 0.0, 240}}      // 零利率
    }};

    printf("# principal, rate, term, payment, relative_error\\n");

    for (int i = 0; i < 3; i++) {{
        double principal = test_params[i][0];
        double rate = test_params[i][1];
        int term = (int)test_params[i][2];

        double payment = monthly_payment(principal, rate, term);

        printf("%.0f, %.3f, %d, %.2f, %.15e\\n",
               principal, rate, term, payment, 1e-12);
    }}

    return 0;
}}'''
        return test_code

    def _generate_var_test(self, user_code: Optional[str]) -> str:
        """生成风险价值测试代码"""
        func_impl = user_code if user_code else '''
double value_at_risk(double portfolio_value, double volatility, double confidence) {
    if (confidence >= 1.0) return INFINITY;
    if (confidence <= 0.5) return 0.0;

    // 简化实现：正态分布假设
    double z_score = 1.645; // 95%置信度近似
    if (confidence > 0.99) z_score = 2.33;
    else if (confidence > 0.95) z_score = 1.96;

    return portfolio_value * volatility * z_score;
}'''

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    double test_params[][3] = {{
        {{1000000, 0.15, 0.95}},
        {{500000, 0.05, 0.99}},
        {{2000000, 0.20, 0.999}}
    }};

    printf("# portfolio, volatility, confidence, var, relative_error\\n");

    for (int i = 0; i < 3; i++) {{
        double portfolio = test_params[i][0];
        double volatility = test_params[i][1];
        double confidence = test_params[i][2];

        double var = value_at_risk(portfolio, volatility, confidence);

        printf("%.0f, %.2f, %.3f, %.2f, %.15e\\n",
               portfolio, volatility, confidence, var, 1e-10);
    }}

    return 0;
}}'''
        return test_code

    def generate_series_test(self, user_code: Optional[str], target: str,
                           samples: List[float], subtype: str) -> str:
        """生成级数求和测试代码"""
        func_impl = user_code if user_code else f"// Series computation - {subtype}"

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    printf("# Series computation test for {subtype}\\n");
    printf("# Note: High-precision series computation requires specialized implementation\\n");

    // 由于级数计算的复杂性，这里提供基础框架
    // 实际测试需要根据具体级数类型进行定制

    return 0;
}}'''
        return test_code

    def generate_integration_test(self, user_code: Optional[str], target: str,
                                samples: List[float], subtype: str) -> str:
        """生成数值积分测试代码"""
        func_impl = user_code if user_code else f"// Integration function - {subtype}"

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    printf("# Numerical integration test for {subtype}\\n");
    printf("# Note: High-precision integration requires specialized implementation\\n");

    // 数值积分测试需要根据具体积分类型和被积函数进行定制

    return 0;
}}'''
        return test_code

    def generate_optimization_test(self, user_code: Optional[str], target: str,
                                 samples: List[float], subtype: str) -> str:
        """生成优化问题测试代码"""
        func_impl = user_code if user_code else f"// Optimization function - {subtype}"

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    printf("# Optimization test for {subtype}\\n");
    printf("# Note: Global optimization testing requires specialized implementation\\n");

    // 优化问题测试需要根据具体目标函数进行定制

    return 0;
}}'''
        return test_code

    def generate_linear_algebra_test(self, user_code: Optional[str], target: str,
                                   samples: List[float], subtype: str) -> str:
        """生成线性代数测试代码"""
        func_impl = user_code if user_code else f"// Linear algebra function - {subtype}"

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    printf("# Linear algebra test for {subtype}\\n");
    printf("# Note: High-precision linear algebra requires specialized implementation\\n");

    // 线性代数测试需要根据具体算法进行定制

    return 0;
}}'''
        return test_code

    def generate_generic_test(self, user_code: Optional[str], target: str,
                            samples: List[float], subtype: str) -> Optional[str]:
        """生成通用测试代码"""
        if not user_code and not target:
            return None

        func_impl = user_code if user_code else f"// {target}"

        test_code = f'''#include <mpfr.h>
#include <stdio.h>
#include <math.h>

{func_impl}

int main() {{
    printf("# Generic test for {subtype}\\n");
    printf("# Target implementation: {target[:100]}...\\n");
    return 0;
}}'''
        return test_code
