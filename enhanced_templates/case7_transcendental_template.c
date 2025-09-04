
#include <mpfr.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


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


int main() {
    // 测试案例 7: tanh_optimized
    printf("Testing tanh_optimized\n");

    // 这里会添加具体的测试逻辑
    return 0;
}
