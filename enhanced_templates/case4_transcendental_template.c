
#include <mpfr.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


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


int main() {
    // 测试案例 4: log1p_optimized
    printf("Testing log1p_optimized\n");

    // 这里会添加具体的测试逻辑
    return 0;
}
