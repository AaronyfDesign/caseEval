
#include <math.h>
#include <float.h>

// 计算NPV及其导数
static void npv_and_derivative(double* cash_flows, int n_periods, double rate,
                              double* npv, double* derivative) {
    *npv = 0.0;
    *derivative = 0.0;

    // 小利率使用泰勒展开
    if (fabs(rate) < 0.005) {
        for (int t = 0; t < n_periods; t++) {
            if (t == 0) {
                *npv += cash_flows[t];
                // t=0项对利率的导数为0
            } else {
                // (1+r)^(-t) ≈ 1 - t*r + t*(t+1)/2*r^2 - ...
                double rt = rate * t;
                double rt2 = rt * rate;
                double factor = 1.0 - rt + 0.5 * t * (t + 1) * rt2;
                double dfactor = -t + t * (t + 1) * rate;

                *npv += cash_flows[t] * factor;
                *derivative += cash_flows[t] * dfactor;
            }
        }
    } else {
        // 标准计算
        for (int t = 0; t < n_periods; t++) {
            if (t == 0) {
                *npv += cash_flows[t];
            } else {
                double factor = pow(1.0 + rate, -t);
                *npv += cash_flows[t] * factor;
                *derivative += cash_flows[t] * (-t) * pow(1.0 + rate, -t - 1);
            }
        }
    }
}

// 计算现金流符号变化次数
static int count_sign_changes(double* cash_flows, int n_periods) {
    int changes = 0;
    int last_sign = 0;

    for (int i = 0; i < n_periods; i++) {
        int current_sign = (cash_flows[i] > 0) ? 1 : ((cash_flows[i] < 0) ? -1 : 0);
        if (current_sign != 0 && last_sign != 0 && current_sign != last_sign) {
            changes++;
        }
        if (current_sign != 0) {
            last_sign = current_sign;
        }
    }
    return changes;
}

// 智能初始估计
static double intelligent_initial_estimate(double* cash_flows, int n_periods) {
    if (n_periods < 2) return 0.1;

    // 简单估计：基于最终值与初始值的比率
    double initial_investment = -cash_flows[0];  // 假设为负值
    double total_inflows = 0.0;

    for (int i = 1; i < n_periods; i++) {
        if (cash_flows[i] > 0) {
            total_inflows += cash_flows[i];
        }
    }

    if (initial_investment > 0 && total_inflows > 0) {
        double total_return = total_inflows / initial_investment;
        double rate_estimate = pow(total_return, 1.0 / (n_periods - 1)) - 1.0;
        return fmax(fmin(rate_estimate, 3.0), -0.99);  // 限制在合理范围
    }

    return 0.1;  // 默认10%
}

double internal_rate_of_return(double* cash_flows, int n_periods) {
    const int MAX_ITERATIONS = 20;
    const double TOLERANCE = 1e-8;

    if (n_periods < 2) return NAN;

    // 检查符号变化
    int sign_changes = count_sign_changes(cash_flows, n_periods);
    bool use_hybrid = (sign_changes > 3);

    // 智能初始估计
    double rate = intelligent_initial_estimate(cash_flows, n_periods);

    if (use_hybrid) {
        // 使用混合bisection-Newton方法
        double rate_low = -0.99, rate_high = 5.0;
        double npv_low, npv_high, dummy;

        npv_and_derivative(cash_flows, n_periods, rate_low, &npv_low, &dummy);
        npv_and_derivative(cash_flows, n_periods, rate_high, &npv_high, &dummy);

        // 确保根被包围
        if (npv_low * npv_high > 0) {
            // 尝试扩展搜索范围
            rate_high = 10.0;
            npv_and_derivative(cash_flows, n_periods, rate_high, &npv_high, &dummy);
        }

        for (int i = 0; i < MAX_ITERATIONS; i++) {
            double npv, derivative;
            npv_and_derivative(cash_flows, n_periods, rate, &npv, &derivative);

            if (fabs(npv) < TOLERANCE) {
                return rate;
            }

            // 检查是否使用Newton步骤
            if (fabs(derivative) > 1e-10) {
                double newton_rate = rate - npv / derivative;

                // 如果Newton步骤在区间内，则使用它
                if (newton_rate > rate_low && newton_rate < rate_high) {
                    rate = newton_rate;
                } else {
                    // 否则使用bisection
                    double npv_current;
                    npv_and_derivative(cash_flows, n_periods, rate, &npv_current, &dummy);

                    if (npv_current * npv_low < 0) {
                        rate_high = rate;
                    } else {
                        rate_low = rate;
                    }
                    rate = (rate_low + rate_high) / 2.0;
                }
            } else {
                // 导数太小，使用bisection
                rate = (rate_low + rate_high) / 2.0;
            }
        }
    } else {
        // 使用纯Newton-Raphson方法
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            double npv, derivative;
            npv_and_derivative(cash_flows, n_periods, rate, &npv, &derivative);

            if (fabs(npv) < TOLERANCE) {
                return rate;
            }

            if (fabs(derivative) < 1e-15) {
                break;  // 导数太小，无法继续
            }

            double delta = npv / derivative;
            rate -= delta;

            // 限制在合理范围内
            rate = fmax(fmin(rate, 5.0), -0.99);

            if (fabs(delta) < TOLERANCE) {
                return rate;
            }
        }
    }

    return rate;  // 返回最后的估计值
}
