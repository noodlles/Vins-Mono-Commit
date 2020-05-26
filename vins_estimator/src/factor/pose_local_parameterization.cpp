#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    // 利用数组来构造Eigen，学习一下（这是取指针吧）
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    // deltaQ:  四元数更新量（输入轴角变化量，输出更新四元数 delta值）
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    // 取数组的指针，构造Eigen
    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    // 状态更新
    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
