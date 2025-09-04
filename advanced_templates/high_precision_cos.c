
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 高精度π/2常数
static const double PI_HALF_HIGH_PREC = 1.5707963267948966192313216916397514;

double high_precision_cos(double x) {
    // 输入范围检查 [1.57079, 1.57081]
    if (x < 1.57079 || x > 1.57081) {
        return NAN;  // 超出定义域
    }

    // 计算相对于π/2的偏移
    double delta = x - PI_HALF_HIGH_PREC;

    // 对于极小的偏移，使用泰勒展开避免相消误差
    // cos(π/2 + δ) = -sin(δ) ≈ -δ + δ³/6 - δ⁵/120 + δ⁷/5040

    if (fabs(delta) < 1e-5) {
        double delta2 = delta * delta;
        double delta3 = delta2 * delta;
        double delta5 = delta3 * delta2;
        double delta7 = delta5 * delta2;

        // 使用泰勒级数（注意符号）
        double result = -delta + delta3 / 6.0 - delta5 / 120.0 + delta7 / 5040.0;

        // 处理次正规数
        if (fabs(result) < DBL_MIN) {
            if (result > 0) return DBL_MIN;
            if (result < 0) return -DBL_MIN;
            return result;  // 可能是真正的零
        }

        return result;
    }

    // 对于较大的偏移，使用标准余弦函数
    // 但由于我们在π/2附近，仍然需要小心处理精度
    return cos(x);
}
