
#include <mpfr.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


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


int main() {
    // 测试案例 9: haversine_distance
    printf("Testing haversine_distance\n");

    // 这里会添加具体的测试逻辑
    return 0;
}
