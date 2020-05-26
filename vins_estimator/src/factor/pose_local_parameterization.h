#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    // 更新状态变量
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    // 由于参数的自由度分为Global和Local，根据链式法则，可以发现，雅克比分为两部分，
    // 第一部分是残差块对Global参数的雅克比，第二部分是Global参数对Local的雅克比
    // 这里计算的就是第二部分：也就是Global参数对Local参数的偏导
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    // 表示参数 xxx 的自由度（可能有冗余），比如四元数的自由度是4，旋转矩阵的自由度是9
    // 这是可能有冗余的自由度数量，在这里是: [x,y,z,qx,qy,qz,qw]
    virtual int GlobalSize() const { return 7; };
    // 这是真正的自由度数量
    // 表示 Δx 所在的正切空间（tangent space）的自由度，这里是 [x,y,z,rx,ry,rz]
    virtual int LocalSize() const { return 6; };
};
