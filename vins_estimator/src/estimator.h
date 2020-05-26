#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>

/**
* @class Estimator 状态估计器
* @Description IMU预积分，图像IMU融合的初始化和状态估计，重定位
* detailed
*/
class Estimator
{
  public:
    Estimator();

    void setParameter();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header);
    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r);

    // internal
    void clearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void solveOdometry();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();


    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag  marginalization_flag;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];           //旋转外参，从相机到imu的旋转
    Vector3d tic[NUM_OF_CAM];           //imu和相机的平移外参,从相机到imu的平移变换

    //窗口中的[P,V,R,Ba,Bg]
    Vector3d Ps[(WINDOW_SIZE + 1)];     //机体在世界坐标系的坐标，机体坐标系到世界坐标系的平移变换
    Vector3d Vs[(WINDOW_SIZE + 1)];     //速度
    Matrix3d Rs[(WINDOW_SIZE + 1)];     //姿态（机体坐标系到世界坐标系的旋转）
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td;

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    std_msgs::Header Headers[(WINDOW_SIZE + 1)];    //滑动窗口内的帧信息

    // 预积分
    // 索引从1开始， 0没有东西
    // pre_integrations[i] 表示的是 从第i-1帧到第i帧的预积分
    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];
    Vector3d acc_0, gyr_0;

    //窗口中的dt,a,v
    vector<double> dt_buf[(WINDOW_SIZE + 1)];                       // 第i-1帧到第i帧之间的imu数据的时间间隔dt集合
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;


    // para_Pose[WINDOW_SIZE]: 最新帧
    // para_Pose[WINDOW_SIZE - 1]: 次新帧
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];               // 11 x 7
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];     // 11 x 9
    double para_Feature[NUM_OF_F][SIZE_FEATURE];                // 1000(应该是会变化的) x 1
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];                 // 单目 : 1 x 7
    double para_Retrive_Pose[SIZE_POSE];                        // 7
    double para_Td[1][1];
    double para_Tr[1][1];

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;             // 上一次边缘化的管理器
    /// marg的时候，得到的要保留的变量的地址
    /// 要保留的变量地址: (指向para_Pose[]..内元素的指针)
    vector<double *> last_marginalization_parameter_blocks;     // 指针容器，储存para_Pose[]...等参数数组内的各个元素地址

    // <kay是时间戳，val是ImageFrame>
    // 图像帧中保存了图像帧的特征点、时间戳、位姿Rt，预积分对象pre_integration，是否是关键帧。
    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration;

    //relocalization variable
    //重定位所需的变量
    bool relocalization_info;
    double relo_frame_stamp;
    double relo_frame_index;
    int relo_frame_local_index;
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE];
    Matrix3d drift_correct_r;
    Vector3d drift_correct_t;
    Vector3d prev_relo_t;           // 闭环帧优化之前的平移
    Matrix3d prev_relo_r;           // 闭环帧优化之前的旋转量
    Vector3d relo_relative_t;
    Quaterniond relo_relative_q;
    double relo_relative_yaw;
};
