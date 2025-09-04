
#include <math.h>
#include <float.h>

// 标准正态分布累积函数
static double norm_cdf(double x) {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}

// 标准正态分布概率密度函数
static double norm_pdf(double x) {
    return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
}

// Black-Scholes期权定价和vega计算
static void black_scholes_call(double S, double K, double T, double r, double sigma,
                               double *price, double *vega) {
    if (T <= 0.0 || sigma <= 0.0) {
        *price = fmax(S - K * exp(-r * T), 0.0);
        *vega = 0.0;
        return;
    }

    double sqrt_T = sqrt(T);
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;

    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    double pdf_d1 = norm_pdf(d1);

    *price = S * Nd1 - K * exp(-r * T) * Nd2;
    *vega = S * pdf_d1 * sqrt_T;
}

// Brenner-Subrahmanyam近似（用于初始猜测）
static double brenner_subrahmanyam_approx(double S, double K, double T, double r, double option_price) {
    double F = S * exp(r * T);  // 远期价格
    double moneyness = fabs(log(F / K));

    if (moneyness < 0.01) {  // ATM情况
        return option_price / (0.398 * S * sqrt(T));
    }

    // 非ATM情况使用修正公式
    double sigma_approx = sqrt(2.0 * M_PI / T) * option_price / S;
    return fmax(sigma_approx, 0.01);
}

double implied_volatility(double S, double K, double T, double r, double option_price) {
    const double MAX_ITERATIONS = 10;
    const double TOLERANCE = 1e-8;
    const double MIN_VEGA = 1e-10;

    // 边界检查
    if (T <= 0.0 || S <= 0.0 || K <= 0.0) return NAN;

    double intrinsic = fmax(S - K * exp(-r * T), 0.0);
    if (option_price < intrinsic) return NAN;

    // ATM稳定性检查
    double moneyness_ratio = S / K;
    bool is_atm = fabs(moneyness_ratio - 1.0) < 0.0001;

    // 初始猜测
    double sigma;
    if (is_atm && T < 0.1) {
        // 短期ATM期权使用Brenner-Subrahmanyam
        sigma = brenner_subrahmanyam_approx(S, K, T, r, option_price);
    } else {
        // 其他情况使用标准初始值
        sigma = 0.2 + 0.1 * fabs(log(moneyness_ratio));
    }

    sigma = fmax(fmin(sigma, 5.0), 1e-6);  // 限制在合理范围内

    // Newton-Raphson迭代
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        double price, vega;
        black_scholes_call(S, K, T, r, sigma, &price, &vega);

        double diff = price - option_price;

        // 收敛检查
        if (fabs(diff) < TOLERANCE) {
            return sigma;
        }

        // Vega除零保护
        if (vega < MIN_VEGA) {
            // 切换到Brent方法或返回当前估计
            break;
        }

        // Newton-Raphson步骤
        double delta_sigma = -diff / vega;

        // 步长限制防止发散
        delta_sigma = fmax(fmin(delta_sigma, 0.5), -0.5);

        sigma += delta_sigma;
        sigma = fmax(fmin(sigma, 5.0), 1e-6);

        // 检查步长是否足够小
        if (fabs(delta_sigma) < TOLERANCE) {
            return sigma;
        }
    }

    return sigma;  // 如果未收敛，返回最后的估计值
}
