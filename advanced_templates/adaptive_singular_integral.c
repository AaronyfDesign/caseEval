
#include <math.h>
#include <quadmath.h>
#include <stdio.h>
#include <stdlib.h>

// 双指数变换处理左端奇点
static double de_transform_left(double (*f)(double), double a, double b, double h) {
    double sum = 0.0;
    int n = 0;

    while (n < 1000) {  // 最大变换点数
        double t = n * h;

        // x = a + (b-a) * exp(-exp(-t))
        double exp_neg_t = exp(-t);
        double exp_neg_exp_neg_t = exp(-exp_neg_t);
        double x = a + (b - a) * exp_neg_exp_neg_t;

        if (x >= b - 1e-15) break;  // 达到右端点

        // 雅可比行列式
        double jacobian = (b - a) * exp_neg_exp_neg_t * exp_neg_t;

        double fx = f(x);
        double weight = jacobian;

        sum += fx * weight * h;

        n++;
    }

    return sum;
}

// 峰值区域高密度采样（FP128精度）
static __float128 peak_region_integrate_fp128(double (*f)(double),
                                              double a, double b, int n_points) {
    __float128 sum = 0.0q;
    double h = (b - a) / n_points;

    // 使用Simpson规则的复合版本
    for (int i = 0; i <= n_points; i++) {
        double x = a + i * h;
        double fx = f(x);
        __float128 weight;

        if (i == 0 || i == n_points) {
            weight = 1.0q;
        } else if (i % 2 == 1) {
            weight = 4.0q;
        } else {
            weight = 2.0q;
        }

        sum += weight * (__float128)fx;
    }

    return sum * (__float128)h / 3.0q;
}

// 检测峰值区域
static int detect_peak_region(double (*f)(double), double x_center, double tolerance) {
    double f_center = f(x_center);
    double f_left = f(x_center - tolerance);
    double f_right = f(x_center + tolerance);

    // 简单的峰值检测：中心值明显大于两侧
    return (f_center > 2.0 * f_left && f_center > 2.0 * f_right);
}

double adaptive_singular_integral(double (*f)(double), double a, double b) {
    const int MAX_FUNCTION_EVALS = 100000000;
    const double TARGET_PRECISION = 1e-30;
    const double PEAK_CENTER = 0.5;
    const double PEAK_HALF_WIDTH = 0.001;  // (0.499, 0.501)

    int function_evals = 0;
    double total_integral = 0.0;

    // 第一部分：[0, 0.499] - 使用双指数变换处理奇点
    if (a < 0.499) {
        double part1_end = fmin(b, 0.499);
        double part1 = de_transform_left(f, a, part1_end, 0.01);
        total_integral += part1;
        function_evals += 1000;  // 估计函数调用次数
    }

    // 第二部分：[0.499, 0.501] - 峰值区域高密度采样
    if (a < 0.501 && b > 0.499) {
        double peak_start = fmax(a, 0.499);
        double peak_end = fmin(b, 0.501);

        // 检测是否确实是峰值区域
        if (detect_peak_region(f, PEAK_CENTER, 0.001)) {
            // 使用5000点/单位宽度
            int n_points = (int)((peak_end - peak_start) * 5000);
            n_points = fmax(n_points, 100);  // 最少100点

            // 使用FP128精度进行积分
            __float128 part2_fp128 = peak_region_integrate_fp128(f, peak_start, peak_end, n_points);
            double part2 = (double)part2_fp128;

            total_integral += part2;
            function_evals += n_points;
        } else {
            // 不是峰值，使用标准积分
            // 这里可以添加标准的自适应积分算法
        }
    }

    // 第三部分：[0.501, 1] - 标准积分方法
    if (b > 0.501) {
        double part3_start = fmax(a, 0.501);
        // 使用自适应Gauss-Kronrod或Simpson方法
        // 这里简化为基本实现
        double part3 = 0.0;  // 需要实现标准积分算法
        total_integral += part3;
    }

    return total_integral;
}

// 示例：被积函数 f(x) = x^(-0.9) / (1 + 1000*(x-0.5)^2)
double example_integrand(double x) {
    if (x <= 0.0) return 0.0;  // 避免奇点

    double x_minus_half = x - 0.5;
    double denominator = 1.0 + 1000.0 * x_minus_half * x_minus_half;

    return pow(x, -0.9) / denominator;
}
