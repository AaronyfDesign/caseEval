
#include <math.h>
#include <float.h>

// 高精度逆误差函数实现（渐近展开）
static double erf_inv_asymptotic(double p) {
    // 对于 p > 0.995 使用渐近展开
    if (p >= 1.0) return INFINITY;
    if (p <= 0.0) return -INFINITY;

    double x = 2.0 * p - 1.0;  // 转换到 [-1, 1]

    if (fabs(x) < 0.99) {
        // 标准多项式近似
        double t = fmin(2.0 - fabs(x), fabs(x));
        double u = t * t - 3.0;
        double z = t * ((0.000345 * u + 0.0209481) * u + 0.07636871) /
                      (((0.005538 * u + 0.128469) * u + 0.555028) * u + 1.0);
        return (x < 0) ? -z : z;
    } else {
        // 高置信度渐近展开
        double t = sqrt(-2.0 * log(1.0 - fabs(x)));
        double z = t - (0.010328 * t + 0.802853) * t /
                      ((0.189269 * t + 1.189715) * t + 1.0);
        return (x < 0) ? -z : z;
    }
}

// Kahan补偿乘法（用于低波动率情况）
static double kahan_multiply(double a, double b, double c) {
    // 计算 a * b * c 时减少舍入误差
    double product1 = a * b;
    double error1 = fma(a, b, -product1);  // 使用FMA获得精确误差

    double product2 = product1 * c;
    double error2 = fma(product1, c, -product2);

    return product2 + error1 * c + error2;
}

double value_at_risk(double portfolio_value, double volatility, double confidence) {
    // 输入验证
    if (portfolio_value < 1000.0 || portfolio_value > 1e9) return NAN;
    if (volatility < 0.01 || volatility > 0.5) return NAN;
    if (confidence < 0.9 || confidence > 0.9999) return NAN;

    // 特殊情况处理
    if (confidence >= 1.0) return INFINITY;

    // 计算标准正态分位数
    double z_score;

    if (confidence > 0.995) {
        // 高置信度使用渐近展开
        z_score = erf_inv_asymptotic(confidence) * M_SQRT2;
    } else {
        // 标准置信度使用多项式近似
        double alpha = 1.0 - confidence;
        double t = sqrt(-2.0 * log(alpha));
        z_score = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) /
                     (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
    }

    // VaR计算
    double var;

    if (volatility < 0.05) {
        // 低波动率使用Kahan补偿乘法
        var = kahan_multiply(portfolio_value, volatility, z_score);
    } else {
        // 标准计算
        var = portfolio_value * volatility * z_score;
    }

    return var;
}
