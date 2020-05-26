#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <fstream>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

// 视觉/IMU边缘化因子，都使用这个类
struct ResidualBlockInfo
{

    ResidualBlockInfo(ceres::CostFunction *_cost_function,          // cost func
                      ceres::LossFunction *_loss_function,          // 核函数
                      std::vector<double *> _parameter_blocks,      // 参数变量(传进来的ceres保持一致，这样才能根据上面的cost func来取变量维度数)
                      std::vector<int> _drop_set)                   // 待marg的变量的序号 (只有两个)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    /// 三种情况的 "parameter_blocks" 内容
    /// 视觉: [Ti, Tj, Tbc , 特征点逆深度]
    /// IMU: [滑动窗口第0帧IMU位姿， 第0帧速度\ba\bg , 滑动窗口第1帧IMU位姿， 第1帧速度\ba\bg  ]
    /// 上一次marg得到的: [ 上一次marg剩下的参数变量 ]
    std::vector<double *> parameter_blocks;         // 参数变量
    std::vector<int> drop_set;                      // 待边缘化的优化变量（数组的起始地址）指针

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;//残差 IMU:15X1 视觉2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

///边缘化的大管家
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;//所有观测项
    // m: 需要marg掉的变量的总维度(local_size)
    // n: 需要保留的变量的总维度数(local_size)
    int m, n;
    std::unordered_map<long, int> parameter_block_size; //global size , <参数变量内存地址(long), 变量的global Size (如7或者9) >
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size , <参数变量内存地址(long), 基于local size 的参数变量索引(如 0,6,15,...)>
    std::unordered_map<long, double *> parameter_block_data;//  <参数变量内存地址(long), 参数变量的真正地址>
    ///[注意] addr(long) 与 地址 的区别:
    ///      addr是地址的long类型， 如果某个参数变量i地址为 "0x7fb44c266570" ，那么 addr=10328408
    // <addr, 参数变量i的地址 (数组的第一个元素地址)数组>
    // 举例 parameter_block_data中的某条映射为: <10328408 , data[]={0x7fb44c266570}>

    // 要保留的变量的global Size (如7或者9)
    std::vector<int> keep_block_size; //（每个变量的global维度数）
    // 要保留的变量的 基于local size 的参数变量索引(如 0+m,6+m,15+m,...)
    std::vector<int> keep_block_idx;  //（每个变量第0个元素在H矩阵中的索引）
    std::vector<double *> keep_block_data;  // 储存要保留变量的 数据

    // 最终的边缘化的结果（数据）
    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

// 继承ceres::CostFunction
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
