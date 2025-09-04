
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const double EARTH_RADIUS = 6371000.0;  // 地球半径（米）
static const double DEG_TO_RAD = M_PI / 180.0;
static const double POLAR_THRESHOLD = 89.0;    // 极地保护阈值

// 规范化经度差
static double normalize_longitude_diff(double dlon) {
    while (dlon > 180.0) dlon -= 360.0;
    while (dlon < -180.0) dlon += 360.0;
    return dlon;
}

// 短距离小角度近似
static double small_angle_distance(double lat1_rad, double lon1_rad,
                                  double lat2_rad, double lon2_rad) {
    double dlat = lat2_rad - lat1_rad;
    double dlon = lon2_rad - lon1_rad;
    double avg_lat = (lat1_rad + lat2_rad) / 2.0;

    // 考虑纬度的经度缩放
    double x = dlon * cos(avg_lat);
    double y = dlat;

    // 弦长近似
    double chord_length = sqrt(x * x + y * y);
    return EARTH_RADIUS * chord_length;
}

// 标准Haversine公式
static double standard_haversine(double lat1_rad, double lon1_rad,
                               double lat2_rad, double lon2_rad) {
    double dlat = lat2_rad - lat1_rad;
    double dlon = lon2_rad - lon1_rad;

    double a = sin(dlat / 2.0) * sin(dlat / 2.0) +
               cos(lat1_rad) * cos(lat2_rad) *
               sin(dlon / 2.0) * sin(dlon / 2.0);

    double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
    return EARTH_RADIUS * c;
}

// 极地区域使用球面余弦定律
static double polar_distance(double lat1_rad, double lon1_rad,
                            double lat2_rad, double lon2_rad) {
    double dlon = lon2_rad - lon1_rad;

    double cos_c = sin(lat1_rad) * sin(lat2_rad) +
                   cos(lat1_rad) * cos(lat2_rad) * cos(dlon);

    // 处理数值误差
    cos_c = fmax(fmin(cos_c, 1.0), -1.0);

    double c = acos(cos_c);
    return EARTH_RADIUS * c;
}

double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    // 输入验证
    if (!isfinite(lat1) || !isfinite(lon1) || !isfinite(lat2) || !isfinite(lon2)) {
        return NAN;
    }

    // 限制纬度范围
    lat1 = fmax(fmin(lat1, 90.0), -90.0);
    lat2 = fmax(fmin(lat2, 90.0), -90.0);

    // 转换为弧度
    double lat1_rad = lat1 * DEG_TO_RAD;
    double lat2_rad = lat2 * DEG_TO_RAD;
    double lon1_rad = lon1 * DEG_TO_RAD;
    double lon2_rad = lon2 * DEG_TO_RAD;

    // 规范化经度差
    double dlon_deg = normalize_longitude_diff(lon2 - lon1);
    double dlon_rad = dlon_deg * DEG_TO_RAD;

    // 更新经度弧度值
    lon2_rad = lon1_rad + dlon_rad;

    // 极地保护
    bool near_pole = (fabs(lat1) > POLAR_THRESHOLD || fabs(lat2) > POLAR_THRESHOLD);

    // 对径点检查
    bool antipodal = (fabs(fabs(dlon_deg) - 180.0) < 0.1);

    if (near_pole || antipodal) {
        return polar_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad);
    }

    // 短距离优化
    double rough_distance = fabs(lat2 - lat1) * EARTH_RADIUS * DEG_TO_RAD +
                           fabs(dlon_deg) * cos((lat1_rad + lat2_rad) / 2.0) * EARTH_RADIUS * DEG_TO_RAD;

    if (rough_distance < 100.0) {  // 小于100米使用小角度近似
        return small_angle_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad);
    }

    // 标准Haversine计算
    return standard_haversine(lat1_rad, lon1_rad, lat2_rad, lon2_rad);
}
