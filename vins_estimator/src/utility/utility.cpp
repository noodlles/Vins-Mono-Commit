#include "utility.h"

// 根据参考坐标系C0下的重力向量，
// 求出 从 参考坐标系C0 到 世界坐标系w_1 的旋转
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    // 求 参考坐标系重力向量 到 (0,0,1)的旋转
    // 也就是 参考坐标系C0到 世界坐标系w_1 的旋转
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    // 取 参考坐标系C0到 世界坐标系w_1 的yaw角
    double yaw = Utility::R2ypr(R0).x();
    // 得到 R0： 参考坐标系C0到 世界坐标系w_1 (不含yaw角)的旋转
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
