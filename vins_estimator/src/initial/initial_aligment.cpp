#include "initial_alignment.h"

/**
 * @brief   陀螺仪偏置校正
 * @optional    根据视觉SFM的结果来校正陀螺仪Bias -> Paper V-B-1
 *              主要是将相邻帧之间SFM求解出来的旋转矩阵与IMU预积分的旋转量对齐
 *              注意得到了新的Bias后对应的预积分需要repropagate
 * @param[in]   all_image_frame所有图像帧构成的map,图像帧保存了位姿、预积分量和关于角点的信息
 * @param[out]  Bgs 陀螺仪偏置
 * @return      void
*/
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 遍历所有图像帧， 取相邻的两帧
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        // 相邻帧j
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();

        //R_ij = (R^c0_bk)^-1 * (R^c0_bk+1) 转换为四元数 q_ij = (q^c0_bk)^-1 * (q^c0_bk+1)
        // 第j帧IMU坐标系 到 第i帧IMU坐标系的旋转
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        /// 要求解的是bg ， 这里的目标函数刚好是 预积分误差中的旋转部分
        /// 所以，雅克比J可以直接取
        /// 所以，这里要构造的最小二乘问题 HX=b  ===> J^T J dx = J^T r
        //tmp_A = J_j_bw
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        //tmp_b = 2 * (r^bk_bk+1)^-1 * (q^c0_bk)^-1 * (q^c0_bk+1)
        //      = 2 * (r^bk_bk+1)^-1 * q_ij
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();  // 与预积分误差形式一致
        //tmp_A * delta_bg = tmp_b
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;

    }
    // LDLT方法
    delta_bg = A.ldlt().solve(b);
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    // 更新滑动窗口中的bg
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;     // 这里的Bgs[] 实际上就是滑动窗口的变量 estimator.Bgs[]

    // 对所有帧进行预积分的重传播
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

//在半径为G的半球找到切面的一对正交基 -> Algorithm 1
MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    // 归一化
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    // 如果前面得到的g0完全只有z轴的分量，那么 tmp=(1,0,0)
    if(a == tmp)
        tmp << 1, 0, 0;
    // 否则tmp=(0,0,1)

    // tmp=(1,0,0) a=[0,0,1]^T
    // b = a.cross(tmp);   // qpc:test
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    // 在bc第3维，分别设置两个正交基
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @brief   重力矢量细化
 * @optional    重力细化，在其切线空间上用两个变量重新参数化重力 -> Paper V-B-3
                g^ = ||g|| * (g^-) + w1b1 + w2b2
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、二自由度重力参数w[w1,w2]^T、尺度s
 * @return      void
*/
void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // g0 = (g^-)*||g||
    Vector3d g0 = g.normalized() * G.norm();    // g0: 方向是"LinearAlignment()"求解得到的 ， 模长(G)是给定的
    Vector3d lx, ly;
    // VectorXd x;
    /// 待优化变量: 滑动窗口每帧速度 v_k^{b_k}，重力向量(2维)，尺度
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;

    for(int k = 0; k < 4; k++)//迭代4次
    {
        //lxly = b = [ 0, 0
        //             0, 0
        //             b1,b2];
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        // 遍历所有图像帧的相邻两帧
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            /// 相邻两帧之间构成的 待优化变量X=[v_k^{bk} , v_{k+1}^{bk+1} , g_^{c0} , s]^T
            /// 残差: 相邻两帧IMU预积分误差（转换到参考坐标系下的预积分误差） [位置预积分误差，速度预积分误差]
            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            // tmp_A(6,9) = [-I*dt           0             (R^bk_c0)*dt*dt*b/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ]
            //              [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt*b                  0                    ]
            // tmp_b(6,1) = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c - (R^bk_c0)*dt*dt*||g||*(g^-)/2 , (b^bk_bk+1)-(R^bk_c0)dt*||g||*(g^-)]^T
            // tmp_A * x = tmp_b 求解最小二乘问题
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0;
            b = b * 1000.0;
            x = A.ldlt().solve(b);
            // 修正重力向量， 这时带上模长
            // dg = [w1,w2]^T
            VectorXd dg = x.segment<2>(n_state - 3);
            g0 = (g0 + lxly * dg).normalized() * G.norm();
            //double s = x(n_state - 1);
    }   
    g = g0;
}

/**
 * @brief   计算尺度，重力加速度和速度
 * @optional    速度、重力向量和尺度初始化Paper -> V-B-2
 *              相邻帧之间的位置和速度与IMU预积分出来的位置和速度对齐，求解最小二乘
 *              重力细化 -> Paper V-B-3
 * @param[in]   all_image_frame 所有图像帧构成的map,图像帧保存了位姿，预积分量和关于角点的信息
 * @param[out]  g 重力加速度
 * @param[out]  x 待优化变量，窗口中每帧的速度V[0:n]、重力g、尺度s
 * @return      void
*/
bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size();
    // 优化量x的总维度
    int n_state = all_frame_count * 3 + 3 + 1;  // 每一帧的IMU坐标系速度， 重力方向向量 ， 尺度

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    // 遍历所有图像帧中的相邻两帧
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        /// 相邻两帧之间构成的 待优化变量X=[v_k^{bk} , v_{k+1}^{bk+1} , g^{c0} , s]^T
        /// 残差: 相邻两帧IMU预积分误差（转换到参考坐标系下的预积分误差） [位置预积分误差，速度预积分误差]

        // J : 6x10
        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        // 残差 r : 6维
        VectorXd tmp_b(6);
        tmp_b.setZero();


        // 取预积分总时间
        double dt = frame_j->second.pre_integration->sum_dt;

        // tmp_A(6,10) = H^bk_bk+1 = [-I*dt           0             (R^bk_c0)*dt*dt/2   (R^bk_c0)*((p^c0_ck+1)-(p^c0_ck))  ]
        //                           [ -I    (R^bk_c0)*(R^c0_bk+1)      (R^bk_c0)*dt                  0                    ]
        // tmp_b(6,1 ) = z^bk_bk+1 = [ (a^bk_bk+1)+(R^bk_c0)*(R^c0_bk+1)*p^b_c-p^b_c , (b^bk_bk+1)]^T  ]
        // Hx=b ===> J^T J * x = J^T r 求解最小二乘问题
        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        /// 这里相当于把尺度 s 乘以100了
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);

    /// 前面构造 J^T 的时候，相当于把s放大了100倍，这里还原
    double s = x(n_state - 1) / 100.0;
    ROS_DEBUG("estimated scale: %f", s);

    g = x.segment<3>(n_state - 4);
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());

    // 检查求出来的重力向量的模长是否偏差太大，但是后面并不需要这个模长，只需要方向即可
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    // 重力细化 （优化方向，然后将模长设置为给定的模长）
    RefineGravity(all_image_frame, g, x);

    // 设置尺度
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());

    if(s < 0.0 )
        return false;   
    else
        return true;
}

/**
 * @brief VisualIMUAlignment
 *        视觉和IMU对齐
 * @param all_image_frame
 * @param Bgs   滑动窗口中的Bgs
 * @param g     重力向量
 * @param x     输出
 * @return
 */
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs); // 计算陀螺仪偏置,然后更新预积分(重传播)

    if(LinearAlignment(all_image_frame, g, x))//计算尺度，重力加速度和速度
        return true;
    else 
        return false;
}
