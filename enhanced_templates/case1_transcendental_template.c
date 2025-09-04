
#include <mpfr.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


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


int main() {
    // 测试案例 1: exp_minus_one
    printf("Testing exp_minus_one\n");

    // 这里会添加具体的测试逻辑
    return 0;
}
