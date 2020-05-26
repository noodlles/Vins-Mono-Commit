#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    clearState();
}

//视觉测量残差的协方差矩阵
void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

//清空或初始化滑动窗口中所有的状态量
void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
        
        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }
    
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }
    
    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }
    
    solver_flag = INITIAL;
    first_imu = false,
            sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;
    
    
    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;
    
    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();
    
    f_manager.clearState();
    
    failure_occur = 0;
    relocalization_info = 0;
    
    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

/**
 * @brief   处理IMU数据
 * @Description IMU预积分，中值积分得到当前PQV作为优化初值
 * @param[in]   dt 时间间隔
 * @param[in]   linear_acceleration 线加速度
 * @param[in]   angular_velocity 角速度
 * @return  void
*/
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    // 检查是不是第一个数据，
    if (!first_imu)
    {
        // 储存第一个数据
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    
    // 检查当前这一帧是否有预积分
    if (!pre_integrations[frame_count])
    {
        // new 一个预积分对象
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    
    // 如果不是第一帧
    if (frame_count != 0)
    {
        // 预积分 (dt,线加速度，角速度)
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); // 临时预积分值,在Estimator::processImage()用到
        
        // 记录数据
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        
        ///???
        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        
        ///由IMU数据得到第j帧IMU位姿估计
        // 采用的是中值积分的传播方式
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;    // 保存k时刻下的加速度  ti<=tk<=tj
    gyr_0 = angular_velocity;
}

/**
 * @brief   处理图像特征数据
 * @Description addFeatureCheckParallax()添加特征点到feature中，计算点跟踪的次数和视差，判断是否是关键帧
 *              判断并进行外参标定
 *              进行视觉惯性联合初始化或基于滑动窗口非线性优化的紧耦合VIO
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   header 某帧图像的头信息
 * @return  void
*/
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    
    // 添加之前检测到的特征点到feature容器中，计算每一个点跟踪的次数，以及它的视差
    // 通过检测两帧之间的视差决定上一帧是否作为关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))  //特征点很少，或者，视差足够，是关键帧
        marginalization_flag = MARGIN_OLD;//=0
    else
        marginalization_flag = MARGIN_SECOND_NEW;//=1
    
    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header;      // 初始化的时候用到，储存时间戳等
    
    // 将图像数据、时间、临时预积分值存到ImageFrame类中
    ImageFrame imageframe(image, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    
    // 容器，储存所有 <时间戳，ImageFrame对象>  (每一帧都储存了，不仅仅是滑动窗口中的帧)
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    
    // 更新临时预积分初始值，准备下一轮的数据
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    
    if(ESTIMATE_EXTRINSIC == 2)//如果没有外参则进行标定
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            //得到两帧之间归一化特征点
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            //标定从camera到IMU之间的旋转矩阵
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }
    
    if (solver_flag == INITIAL)//初始化
    {
        // frame_count是滑动窗口中图像帧的数量，一开始初始化为0，滑动窗口总帧数WINDOW_SIZE=10
        // 确保有足够的frame参与初始化
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            // 有 外参且当前帧时间戳大于初始化时间戳0.1秒，就进行初始化操作
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                // 视觉惯性联合初始化
                result = initialStructure();
                // 更新初始化时间戳
                initial_timestamp = header.stamp.toSec();
            }
            if(result)//初始化成功
            {
                //先进行一次滑动窗口非线性优化，得到当前帧与第一帧的位姿
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
                
            }
            else
                slideWindow();//初始化失败则直接滑动窗口
        }
        else
            frame_count++;//图像帧数量+1
    }
    else//紧耦合的非线性优化
    {
        //==========================Debug START=====================================

        /** ******************************************************
         * debug
         * 打印输出:
         * 1. 初始化完成之后的滑动窗口特征点数量
         * 2. 特征点的start_frame
         * 3. 特征点的共视帧数量
         * *******************************************************/

        //ROS_INFO("//=======================Debug START============================");

        //ROS_INFO("Total Feature: [%d]",f_manager.feature.size());
        //for(auto it : f_manager.feature){
        //    std::cout<<"特征点ID: ["<<it.feature_id<<"] "<<
        //               "第一帧ID: ["<<it.start_frame<<"] "<<
        //               "共视帧数: ["<<it.feature_per_frame.size()<<"] "<<endl;
        //}

        //============================Debug END=====================================

        TicToc t_solve;
        solveOdometry();
        ROS_DEBUG("solver costs: %fms", t_solve.toc());
        
        //故障检测与恢复,一旦检测到故障，系统将切换回初始化阶段
        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }
        
        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);
        
        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

/**
 * @brief   视觉的结构初始化
 * @Description 确保IMU有充分运动激励
 *              relativePose()找到具有足够视差的两帧,由F矩阵恢复R、t作为初始值
 *              sfm.construct() 全局纯视觉SFM 恢复滑动窗口帧的位姿
 *              visualInitialAlign()视觉惯性联合初始化
 * @return  bool true:初始化成功
*/
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    
    // 通过加速度标准差判断IMU是否有充分运动以初始化。
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        // 遍历化的滑动窗口图像
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            // 取该帧预积分总时间
            double dt = frame_it->second.pre_integration->sum_dt;
            // 求加速度
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        // 求平均加速度
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        // 求方差
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));//标准差
        //ROS_WARN("IMU variation %f!", var);
        
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    // 将f_manager中的所有feature保存到存有SFMFeature对象的sfm_f中
    Quaterniond Q[frame_count + 1];         ///窗口内图像帧的旋转四元数q（相对于参考帧[第l帧]，到第l帧的旋转变换）
    Vector3d T[frame_count + 1];            ///窗口内图像帧的平移向量T（相对于参考帧[第l帧]，到第l帧的平移变换）
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    // 遍历特征点
    for (auto &it_per_id : f_manager.feature)
    {
        // 取特帧点起始帧的前一帧
        int imu_j = it_per_id.start_frame - 1;
        // 构造"SFMFeature"类对象
        SFMFeature tmp_feature;
        tmp_feature.state = false;                  // 是否被三角化
        tmp_feature.id = it_per_id.feature_id;      // 特征点id
        // 遍历观测到该特征点的image
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // 取特征点在该帧的归一化平面点
            Vector3d pts_j = it_per_frame.point;
            // 构成pair<帧id，归一化平面点坐标(x,y)>
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    
    // 保证具有足够的视差的帧作为参考帧，然后由F矩阵恢复Rt (得到从 最新帧到参考帧的变换)
    // 第l帧是从第一帧开始到滑动窗口中第一个满足与当前帧的平均视差足够大的帧，会作为参考帧到下面的全局sfm使用
    // 此处的relative_R，relative_T为当前帧到参考帧（第l帧）的变换Rt
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    
    // 对窗口中每个图像帧求解sfm问题
    // 得到所有图像帧相对于参考帧的姿态四元数Q、平移向量T和特征点坐标sfm_tracked_points。
    GlobalSFM sfm;
    // frame_count: 10
    if(!sfm.construct(frame_count + 1, Q, T, l,
                      relative_R, relative_T,
                      sfm_f, sfm_tracked_points))
    {
        // 求解失败则边缘化最早一帧并滑动窗口
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    /** *********************************************************************
     * Debug
     * 1. 打印参考帧(第l帧)序号
     * 2. 打印三角化点数
     * **********************************************************************/
    //std::cout<<"参考帧: ["<<l<<"]  "<<"三角化点数: ["<<sfm_tracked_points.size()<<"] "<<std::endl;

    
    // solve pnp for all frame
    // 对于所有的图像帧，包括不在滑动窗口中的，提供初始的RT估计，然后solvePnP进行优化,得到每一帧的姿态
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)     // 遍历所有图像帧
    {
        // provide initial guess
        // 取初始值
        cv::Mat r, rvec, t, D, tmp_r;
        // Headers[]: 滑动窗口中的帧的信息
        if((frame_it->first) == Headers[i].stamp.toSec())   // 初始化: 滑动窗口内的帧都是关键帧
        {
            // 是关键帧
            // 取BA的优化值，进行设置
            frame_it->second.is_key_frame = true;
            // 转换为 IMU坐标系
            // Q[i].toRotationMatrix() : [相机坐标系]到 参考帧C0[相机坐标系]的变换
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;    // 指向下一帧
            // 关键帧不用参与后面的Pnp求解，直接跳过
            continue;
        }

        /// 非关键帧

        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        
        //Q和T是图像帧的位姿，而不是求解PNP时所用的坐标系变换矩阵
        /// T: 窗口内图像帧 到 第l帧 的平移变换
        /// Q: 窗口内图像帧 到 第l帧 的旋转变换
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        //罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);
        
        frame_it->second.is_key_frame = false;          // 非关键帧
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        // 遍历该帧的特征点
        for (auto &id_pts : frame_it->second.points)
        {
            // 特征点id
            int feature_id = id_pts.first;
            // 遍历该特征点的数据
            for (auto &i_p : id_pts.second)
            {
                // 根据特征点id，检查是否被跟踪的点
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    // 特征点有被跟踪到

                    // 取特征点在SFM时得到的世界坐标(3D)[相对于参考帧的]
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);

                    // i_p的数据内容: <特征点id，(camera_id,[x,y,z,u,v,vx,vy])>
                    // 取归一化平面坐标
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        //保证特征点数大于5
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        /**
         *bool cv::solvePnP(    求解pnp问题
         *   InputArray  objectPoints,   特征点的3D坐标数组
         *   InputArray  imagePoints,    特征点对应的图像坐标
         *   InputArray  cameraMatrix,   相机内参矩阵
         *   InputArray  distCoeffs,     失真系数的输入向量
         *   OutputArray     rvec,       旋转向量 (从参考坐标系 到 某一帧坐标系的变换)
         *   OutputArray     tvec,       平移向量 (从参考坐标系 到 某一帧坐标系的变换)
         *   bool    useExtrinsicGuess = false, 为真则使用提供的初始估计值
         *   int     flags = SOLVEPNP_ITERATIVE 采用LM优化
         *)
         */
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        // 这里也同样需要将坐标变换矩阵转变成图像帧位姿，并转换为IMU坐标系的位姿
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        // frame_it->second.R : 图像帧 [IMU坐标系] 到 参考帧C0 相机坐标系的变换
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    
    //进行视觉惯性联合初始化
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
    
}

/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    // solve scale
    // 计算陀螺仪偏置，尺度，重力加速度和速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }
    
    // change state
    // 得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    // 遍历滑动窗口,为滑动窗口中的帧设置位姿
    for (int i = 0; i <= frame_count; i++)
    {
        // R： 图像帧 IMU坐标系 到 参考帧坐标系的变换
        // T： 图像帧 [相机坐标系] 到 参考帧C0 相机坐标系的平移
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }
    
    // 将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);
    
    // triangulat on cam pose , no tic
    // 重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));
    
    double s = (x.tail<1>())(0);

    // 陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    
    //将Ps按尺度s缩放
    // ????
    for (int i = frame_count; i >= 0; i--){
        // old: Ps[i]  第i帧 相机坐标系 到 参考帧坐标系C0 的平移
        // R_{ref<-Imu}[i] * (RIC*P_in_I+TIC[0]) + t_{ref<-Imu} : 第i帧 相机坐标系 到 参考坐标系的平移
        // s*Ps[i] = R_{ref<-Imu}[i] * (RIC*P_in_I+TIC[0]) + t_{ref<-Imu}
        // ==> t_{ref<-Imu} = s*Ps[i] - R_{ref<-Imu}[i] * (RIC*P_in_I+TIC[0]) <===> 第i帧 IMU坐标系 到 参考坐标系的平移
        /// 这里画图，好理解一些
        /// TIC[0] : (相机坐标系原点 在 IMU坐标系的坐标)
        /// Rs[i] * TIC[0]： 第i帧: 相机坐标系 到 IMU坐标系的平移 (在参考坐标系C0下的表示)
        /// Rs[0] * TIC[0]： 第0帧: 相机坐标系 到 IMU坐标系的平移 (在参考坐标系C0下的表示)
        /// s * Ps[i]: 第i帧相机坐标系到 参考坐标系的平移        (在参考坐标系C0下的表示)
        /// s * Ps[0]: 第0帧相机坐标系到 参考坐标系的平移        (在参考坐标系C0下的表示)
        /// (s * Ps[i] - Rs[i] * TIC[0]): 第i帧 IMU坐标系 到 参考坐标系的平移
        /// (s * Ps[0] - Rs[0] * TIC[0]): 第0帧 IMU坐标系 到 参考坐标系的平移
        /// 所以，Ps[i] : 第i帧IMU坐标系 到 第0帧IMU坐标系的平移 (在参考坐标系C0下的表示)
        Ps[i] = (s * Ps[i] - Rs[i] * TIC[0]) - (s * Ps[0] - Rs[0] * TIC[0]);
    }
    
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            //Vs为优化得到的速度
            // x.segment<3>(kv * 3) 是 v_{k}^{bk} ， v_{k+1}^{bk+1} ...
            // 因此，需要乘以 R， 将速度转化到 参考坐标系C0下
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    
    // 特征点逆深度， 乘以尺度因子
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }
    
    ///参考坐标系C0 到 世界坐标系w_1 的旋转不仅仅是yaw，   第0帧IMU坐标系到世界坐标系w_1的旋转才是只有yaw的差异
    // 通过将重力旋转到z轴上，得到[世界坐标系w_1]与[摄像机坐标系c0]之间的旋转矩阵rot_diff
    // R0： 参考坐标系C0到 世界坐标系w_1 (不含yaw角)的旋转 [因为yaw角不可观？ 下面会把世界坐标系的初始yaw角设置为与滑动窗口第一帧为起始]
    Matrix3d R0 = Utility::g2R(g);
    // Rs[0]: 滑动窗口第0帧 IMU坐标系 到 参考坐标系的旋转
    // R0: 参考坐标系到世界坐标系的旋转
    // yaw: 滑动窗口第0帧 IMU坐标系 到 世界坐标系w_1的 Yaw旋转
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    // R0: R0包含两个部分的旋转，先1后2
    //    1. 重力对齐: [参考坐标系] 到 [世界坐标系w_1] 的旋转
    //    2. yaw对齐到第0帧IMU坐标系: [世界坐标系w_1] 到 [滑动窗口第0帧IMU坐标系] 的 Yaw旋转 ====>(认为世界坐标系w_1与第0帧IMU坐标系仅有yaw的不一样)
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // 将c0(参考坐标系[第l帧相机坐标系])坐标系中的 重力向量 旋转到 第0帧IMU坐标系,得到 g^{w}
    g = R0 * g;
    
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    // R0: 参考坐标系到滑动窗口第0帧IMU坐标系 的 旋转变换, 当然还要考虑平移部分， 不过平移部分在"Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);"处理了
    Matrix3d rot_diff = R0;
    // 所有变量从参考坐标系c0旋转到滑动窗口第0帧IMU坐标系 [即，最终的世界坐标系w_2]
    for (int i = 0; i <= frame_count; i++)
    {
        // old: Ps[] 每一帧IMU坐标系 到 第0帧IMU坐标系的平移 (在参考坐标系C0下的表示)
        // new: Ps[] 每一帧位姿（IMU坐标系）到 第0帧IMU坐标系的平移变换 (在第0帧IMU坐标系下的表示)
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());
    
    return true;
}

/**
 * @brief   判断两帧有足够视差30且内点数目大于12则可进行初始化，同时得到R和T
 * @Description    判断每帧到窗口最后一帧对应特征点的平均视差是否大于30
                solveRelativeRT()通过基础矩阵计算当前帧与第l帧之间的R和T,并判断内点数目是否足够
 * @param[out]   relative_R 当前帧到第l帧之间的旋转矩阵R
 * @param[out]   relative_T 当前帧到第l帧之间的平移向量T
 * @param[out]   L 参考帧: 保存滑动窗口中与当前帧满足初始化条件的那一帧
 * @return  bool 1:可以进行初始化;0:不满足初始化条件
*/
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    // 寻找第i帧到窗口最后一帧的对应特征点
    // 遍历滑动窗口
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // 获取第i帧与最新帧[WINDOW_SIZE]两帧之间的对应(关联)特征点
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        // 关联点>20
        if (corres.size() > 20)
        {
            //计算平均视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                //第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
                
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            
            // 判断是否满足初始化条件：视差>30和内点数满足要求
            // 同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
            // 460是焦距？
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

//三角化求解所有特征点的深度，并进行非线性优化
void Estimator::solveOdometry()
{
    // 如果总帧数还<滑动窗口要求的帧数，直接返回
    if (frame_count < WINDOW_SIZE)
        return;
    // 求解器: 非线性
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        // 进行三角化
        f_manager.triangulate(Ps, tic, ric);
        ROS_DEBUG("triangulation costs %f", t_tri.toc());
        // 进行滑动窗口的非线性优化
        optimization();
    }
}

//vector转换成double数组，因为ceres使用数值数组
//Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();
        
        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();
        
        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();
        
        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    // 获取特征点逆深度
    VectorXd dep = f_manager.getDepthVector();
    // 设置初始值
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    // 如果需要估计 IMU和相机的不同步时间td，那么也设置初始值
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

// 数据转换，vector2double的相反过程
// 同时这里为防止优化结果往零空间变化，会根据优化前后第一帧的位姿差进行修正。
void Estimator::double2vector()
{
    // 窗口第一帧之前的位姿
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);     // 旋转矩阵转欧拉角
    Vector3d origin_P0 = Ps[0];                     // 取第一帧平移量
    
    // 检测是否发生错误？
    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    
    // 窗口第一帧优化后的位姿
    Vector3d origin_R00 = Utility::R2ypr(
                Quaterniond(para_Pose[0][6],
                            para_Pose[0][3],
                            para_Pose[0][4],
                            para_Pose[0][5]).toRotationMatrix());
    // 求得优化前后的姿态差(yaw角)
    // 优化前-优化后
    double y_diff = origin_R0.x() - origin_R00.x();

    //TODO
    // 取优化前后的yaw角差
    // 这是第一帧: 优化前<--优化后 的旋转变换
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    // origin_R0.y()： pitch角
    // pitch 很接近90度了，是一个奇异点，需要使用四元数来求
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        // 使用四元数来求优化前后的姿态差值
        // Rs[0]: 窗口第一帧优化之前的姿态 (机体坐标系到世界坐标系的旋转)
        // Q: 窗口第一帧优化之后的姿态
        // Q.T : (世界坐标系到机体坐标系的旋转变换)
        // rot_diff:
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }
    // 根据位姿差做修正，即保证第一帧优化前后位姿不变
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                para_Pose[i][1] - para_Pose[0][1],
                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;
        
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                para_SpeedBias[i][1],
                para_SpeedBias[i][2]);
        
        Bas[i] = Vector3d(para_SpeedBias[i][3],
                para_SpeedBias[i][4],
                para_SpeedBias[i][5]);
        
        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                para_SpeedBias[i][7],
                para_SpeedBias[i][8]);
    }
    
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                para_Ex_Pose[i][1],
                para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                para_Ex_Pose[i][3],
                para_Ex_Pose[i][4],
                para_Ex_Pose[i][5]).toRotationMatrix();
    }
    
    // 设置优化之后的特征点深度
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    // 不同步时间估计
    if (ESTIMATE_TD)
        td = para_Td[0][0];
    
    // relative info between two loop frame
    if(relocalization_info)
    {
        // 修正回环帧的机体位姿
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                relo_Pose[1] - para_Pose[0][1],
                relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        // 闭环帧： 优化前的yaw角 - 优化后的(修正过的)yaw角
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        // 这是闭环帧: 优化前<--优化后 的旋转变换
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        // 闭环帧： 优化前<--优化后 的平移变换
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;

        ///计算闭环帧与滑动窗口帧之间的相对位姿关系
        //relo_frame_local_index: 滑动窗口中与闭环帧匹配的帧?
        // 滑动窗口帧到闭环帧的平移
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        // 滑动窗口帧到闭环帧的旋转变换 Rrs   relo<---slidewindow
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        // 滑动窗口帧yaw角 - 闭环帧yaw角 ====>  也就是: 闭环帧<--滑动窗口帧 的yaw角旋转变换
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;
        
    }
}

//系统故障检测 -> Paper VI-G
bool Estimator::failureDetection()
{
    //在最新帧中跟踪的特征数小于某一阈值
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    
    //偏置或外部参数估计有较大的变化
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    
    //最近两个估计器输出之间的位置或旋转有较大的不连续性
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}


/**
 * @brief   基于滑动窗口紧耦合的非线性优化，残差项的构造和求解
 * @Description 添加要优化的变量 (p,v,q,ba,bg) 一共15个自由度，IMU的外参也可以加进来
 *              添加残差，残差项分为4块 先验残差+IMU残差+视觉残差+闭环检测残差
 *              根据倒数第二帧是不是关键帧确定边缘化的结果
 * @return      void
*/
void Estimator::optimization()
{
    // 创建ceres::problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);

    /// 对于四元数或者旋转矩阵这种使用过参数化表示旋转的方式，
    /// 它们是不支持广义的加法（因为使用普通的加法就会打破其 constraint，比如旋转矩阵加旋转矩阵得到的就不再是旋转矩阵），
    /// 所以我们在使用ceres对其进行迭代更新的时候就需要自定义其更新方式了，
    /// 具体的做法是实现一个参数本地化的子类，需要继承于LocalParameterization，
    /// LocalParameterization是纯虚类，
    /// 所以我们继承的时候要把所有的纯虚函数都实现一遍才能使用该类生成对象
    /// 更详细的内容，可参考: https://blog.csdn.net/hzwwpgmwy/article/details/86490556

    // 添加ceres参数块
    // 因为ceres用的是double数组，所以在下面用vector2double做类型装换
    // Ps、Rs转变成para_Pose，Vs、Bas、Bgs转变成para_SpeedBias
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // new一个子类对象，指针赋值成ceres::LocalParameterization
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        // P，Q 参数块， Global是7自由度，Local是6自由度
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        // V，ba，bg参数块
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    
    // ESTIMATE_EXTRINSIC!=0则camera到IMU的外参也添加到估计
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        // 外参估计，同样使用自己重载的PoseLocalParameterization类
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    //相机和IMU硬件不同步时估计两者的时间偏差
    if (ESTIMATE_TD)
    {
        problem.AddParameterBlock(para_Td[0], 1);
        //problem.SetParameterBlockConstant(para_Td[0]);
    }
    
    TicToc t_whole, t_prepare;

    // 把一些初始值，从Eigen转到上面的double的参数块中
    vector2double();

    /// 这里还没看
    // 添加边缘化先验的约束
    if (last_marginalization_info)
    {
        // 边缘化先验转成的约束
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        // 添加到problem
        problem.AddResidualBlock(marginalization_factor,                        // cost func
                                 NULL,                                          // 核函数
                                 last_marginalization_parameter_blocks          // 待优化变量(根据上一次marg的时候，得到的要保留的变量的地址)
                                 );
    }
    
    //添加IMU残差
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        // 遍历滑动窗口,取两帧
        int j = i + 1;
        // 预积分时间太长了? 就不作为约束了
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        // 取预积分数据，构造残差项、以及计算雅克比， 就是构造了一个关于预积分的cost funstion
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor,            // 损失函数
                                 NULL,                  // 核函数
                                 para_Pose[i],          // 第i帧 待优化参数 11 x 7
                                 para_SpeedBias[i],     //                11 x 9
                                 para_Pose[j],          // 第j帧 待优化参数
                                 para_SpeedBias[j]);    //
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    
    //添加视觉残差
    for (auto &it_per_id : f_manager.feature)
    {
        // 遍历feature容器
        // 取一个特征点，检查被观测次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 因为不是所有的特征点都被加入，因此需要记录当前索引
        ++feature_index;

        // imu_i: 特征点第一次被观测到的帧号
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        // 取特征点，第一次被观测到的，那一帧上的归一化平面点
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        // 遍历该特征点被观测到的数据
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            // 在第j帧被观测到了， 在第j帧上的 归一化平面坐标为pts_j
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD)
            {
                // 如果估计TD，那么使用这个残差类
                // 构建视觉残差,并且计算雅克比
                ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                        it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                        it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
                /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                // 不估计TD，则使用这个残差类
                // 构造cost func
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f,                     // 视觉误差 cost func
                                         loss_function,         // 核函数
                                         para_Pose[imu_i],      // 待优化参数    第i帧机体位姿
                                         para_Pose[imu_j],      //             第j帧机体位姿
                                         para_Ex_Pose[0],       //             外参
                        para_Feature[feature_index]);           //             特征点逆深度
            }
            f_m_cnt++;      // 统计视觉观测数量
        }
    }
    
    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());
    
    // 添加闭环检测残差，计算滑动窗口中与每一个闭环关键帧的相对位姿，这个相对位置是为后面的图优化准备
    if(relocalization_info)
    {
        //printf("set relocalization factor! \n");

        // 构造重定位位姿参数块
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        // 重定位位姿: relo_Pose : 7维变量
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        int feature_index = -1;
        // 遍历feature容器
        for (auto &it_per_id : f_manager.feature)
        {
            // 取特征点，以及被观测次数
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            // 取该特征点第一次被观测的帧号
            int start = it_per_id.start_frame;
            // 如果第一次被观测的帧号 < 当前需要重定位的帧号
            if(start <= relo_frame_local_index)
            {
                // match_points[retrive_feature_index].z() : 匹配点的id
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id)
                {
                    retrive_feature_index++;
                }
                // 如果找到匹配点了
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id)
                {
                    // 特征点的闭环帧匹配点在闭环帧上的归一化平面坐标
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0);
                    // 特征点在第一次被观测的帧的归一化点坐标
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                    
                    // 构造视觉的cost func
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f,                             // 视觉重投影误差 cost func
                                             loss_function,                 // 核函数
                                             para_Pose[start],              // 第一次观测到该特征点的帧时刻所对应的机体位姿
                                             relo_Pose,                     // 闭环关键帧的机体位姿
                                             para_Ex_Pose[0],               // 外参
                            para_Feature[feature_index]);  // 该特征点的逆深度
                    retrive_feature_index++;
                }
            }
        }
        
    }
    
    ceres::Solver::Options options;
    
    options.linear_solver_type = ceres::DENSE_SCHUR;            // 求解器类型
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;         // 求解方法
    options.max_num_iterations = NUM_ITERATIONS;                // 最大迭代次数
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    // 根据边缘化方式，设置最大求解时间
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    // 开始求解
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());
    
    ///求解完成之后
    // 防止优化结果在零空间变化，通过固定第一帧的位姿
    double2vector();
    
    TicToc t_whole_marginalization;
    
    //边缘化处理
    //如果次新帧是关键帧，将边缘化最老帧，及其看到的路标点和IMU数据，将其转化为先验：
    if (marginalization_flag == MARGIN_OLD)
    {
        ///边缘化的大管家
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        // 前面进行了优化，状态有了变化，这里重新转成 double 数组
        vector2double();
        
        //1、将上一次先验残差项传递给marginalization_info
        if (last_marginalization_info)
        {
            vector<int> drop_set;
            // 遍历上一次边缘化的参数块
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                // 边缘化就是对滑动窗口中最前面的那一帧进行，也就是 para_Pose[0]、para_SpeedBias[0]
                // 这里的作用是: 找到 [第一帧对应的数据] 在上一次边缘化保留的变量中的索引i
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                        last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);  // 只push两次
            }
            // construct new marginlization_factor
            // 接着利用上一次的边缘化结果，构建边缘化因子
            // last_marginalization_parameter_blocks： 参数变量（指针）
            // drop_set： 对应上面的参数变量中，需要被marg掉的变量索引i
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor,                  //
                                                                           NULL,
                                                                           last_marginalization_parameter_blocks,   // 上一帧marg剩下的Hessian
                                                                           drop_set);                               // 待marg的变量的序号 (只有两个)
            
            // 添加到边缘化管理器
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }
        
        //2、将第0帧和第1帧间的IMU因子IMUFactor(pre_integrations[1])，添加到marginalization_info中
        {
            // pre_integrations[0]: 表示滑动窗口的第0帧与前一帧之间的预积分
            // pre_integrations[1]: 这才是滑动窗口第0帧与第1帧之间的预积分
            // 预积分时间不是太长，可作为先验约束
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                // 构造IMU预积分约束
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                // (线性化点的)参数变量： 这里传进去的是 指针集合 {第0帧imu位姿，第0帧(speed,ba,bg)，第1帧imu位姿，第1帧(speed,ba,bg)}
                // vector<int>{0, 1}: 表示上面的参数中，第0个和第1个，也就是{第0帧imu位姿，第0帧(speed,ba,bg)}需要被边缘化掉
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor,
                                                                               NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                // 添加到边缘化管理器
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }
        
        //3、将第一次观测为第0帧的所有路标点对应的视觉观测，添加到marginalization_info中
        {
            int feature_index = -1;
            // 遍历滑动窗口特征点
            for (auto &it_per_id : f_manager.feature)
            {
                // 根据被观测次数筛选
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;
                
                ++feature_index;
                
                // imu_i : 特征点的父帧
                // imu_j : 这里随便设置了个初始值
                // 选出 父帧是第一帧的特征点
                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;
                
                // 特征点i 在 父帧 的归一化平面点
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;
                
                // 遍历所有观测到 特征点i 的帧
                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    // imu_j : 当前观测到 特征点i 的帧号
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;
                    
                    // 第j帧观测到特征点i 的归一化平面坐标
                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},  //(线性化点的)参数变量
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        // 构建视觉残差
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        // 构造视觉边缘化因子
                        // (线性化点的)参数变量： 这里传进去的是 指针集合 {第i帧imu位姿，第j帧imu位姿，外参，当前遍历的这个特征点}
                        // vector<int>{0, 3}： 表示上面的参数变量第0个和第3个 也就是{第i帧imu位姿}和{当前遍历的这个特征点}
                        // 需要被边缘化掉
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f,
                                                                                       loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }
        
        TicToc t_pre_margin;
        
        
        //4、计算每个残差，对应的Jacobian，并更新parameter_block_data(对应参数变量的内存地址"0x...")
        ///  计算残差和雅克比的目的是: 将先验信息转化为约束项
        //   这是因为，使用ceres，我们没有办法像通常的边缘化得到的Hessian和b直接加到全系统的Hessian和b
        //   所以，只能将边缘化得到的Hessian和b转化为约束项，然后添加到ceres中
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        
        //5、多线程构造先验项舒尔补AX=b的结构，在X0处线性化计算Jacobian和残差
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());
        
        //6.调整参数块在下一次窗口中对应的位置（往前移一格），注意这里是指针，后面slideWindow中会赋新值，
        //  举例: 比如marg的时候，marg第0帧，保留第1,2...帧的变量，那么此时得到的要保留的变量地址是"第1,2...帧"的
        //       但是，在下一次优化的时候，原本"第1,2...帧"的变量，要变成"第0,1...帧"的变量
        //       因此，这里提前把 要保留的变量地址 指向了 "第0,1...帧"的变量
        std::unordered_map<long, double *> addr_shift;
        // 遍历保留下来的变量
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            // 前移
            // <第i帧的位姿参数地址(long) , 第i-1帧的位姿参数真正地址>
            // 所以使用addr_shift 检索 第i帧位姿参数地址(long)，最终得到的是第i-1帧的位姿参数真正地址
            // 这是因为，在下一次优化的时候，由于边缘化掉了最old的帧，因此，原本第1帧变成了第0帧...
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }

        // 获取
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
        
        if (last_marginalization_info)
            delete last_marginalization_info;
        // 将边缘化的结果保存到 last_marginalization_info
        last_marginalization_info = marginalization_info;
        // last_marginalization_parameter_blocks
        // 指针容器，储存para_Pose[]...等参数数组内的各个元素地址
        // marg之后，新的参数块，其中有:
        // 1. para_Pose[]
        // 2. para_SpeedBias[]
        // 3. para_Ex_Pose[]
        // 4. para_Td[]
        last_marginalization_parameter_blocks = parameter_blocks;

        /** ******************************************************
         * debug
         * 打印输出:
         * 1. parameter_block_size
         * 2. parameter_block_data
         * 3. parameter_block_idx
         * *******************************************************/
//        std::cout<<"=======================Debug START============================"<<std::endl;

//        std::cout<<"需要marg的变量总维度(local_size): ["<<marginalization_info->m<<"] "<<std::endl;
//        std::cout<<"需要保留的变量总维度(local_size): ["<<marginalization_info->n<<"] "<<std::endl;
//        // 为了方便看，先排序一下
//        std::vector<std::pair<long, int>> tmp_idx;
//        for (auto& i : marginalization_info->parameter_block_idx)
//            tmp_idx.push_back(i);

//        std::sort(tmp_idx.begin(), tmp_idx.end(),
//                  [=](std::pair<long, int>& a, std::pair<long, int>& b) { return a.second < b.second; });

//        int count=0;
//        for(auto it : tmp_idx){
//            long addr_long=it.first;
//            int global_size=marginalization_info->parameter_block_size[addr_long];
//            double * addr_double=marginalization_info->parameter_block_data[addr_long];
//            int idx=it.second;
//            std::cout<<"count: ["<<count<<"] "<<
//                       "addr(long): ["<<addr_long<<"] "<<
//                       "global_size: ["<<global_size<<"] "<<
//                       "addr(double*): ["<<addr_double<<"] "<<
//                       "idx(local_size): ["<<idx<<"] "<<
//                       std::endl;
//            count++;
//        }
        
    }
    //如果次新帧不是关键帧：
    else
    {
        if (last_marginalization_info &&
                std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {
            //1.保留次新帧的IMU测量，丢弃该帧的视觉测量，将上一次先验残差项传递给marginalization_info
            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();

            // 如果不是第一次marg，即有上一次边缘化的结果
            if (last_marginalization_info)
            {
                // 需要被边缘化的变量索引 , 这个索引指的是在待优化参数块
                vector<int> drop_set;
                // last_marginalization_parameter_blocks： 指针容器，储存para_Pose[]...等参数数组内的各个元素地址
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    // 只边缘化视觉部分
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor,
                                                                               NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);
                
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
            
            //2、premargin
            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());
            
            //3、marginalize
            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            //4.调整参数块在下一次窗口中对应的位置（去掉次新帧）
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)   // [WINDOW_SIZE-1]帧=次新帧
                    continue;
                else if (i == WINDOW_SIZE)  // 原来滑窗中的最新帧( 也就是次新帧 ) [要被marg掉的]
                {
                    // marg掉原来滑窗中的次新帧([WINDOW_SIZE-1]帧)，
                    // 在下一次优化的时候，最新帧的数据前移到 [WINDOW_SIZE-1]帧上了
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else    // 其他帧不变
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;

            // 将边缘化的结果保存到 last_marginalization_info
            last_marginalization_info = marginalization_info;
            // 指针容器，储存para_Pose[]...等参数数组内的各个元素地址
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc());
}

/**
 * @brief   实现滑动窗口all_image_frame的函数
 * @Description 如果次新帧是关键帧，则边缘化最老帧，将其看到的特征点和IMU数据转化为先验信息
                如果次新帧不是关键帧，则舍弃视觉测量而保留IMU测量值，从而保证IMU预积分的连贯性
 * @return      void
*/
void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        // 备份
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        // 最新帧[WINDOW_SIZE]
        if (frame_count == WINDOW_SIZE)
        {
            /// 遍历滑动窗口,进行数据交换
            /// 目标：把滑动窗口最前面的数据移到最后
            ///      实现marg之后，要保留的数据与指针对齐(对应MarginalizationInfo::getParameterBlocks()函数的操作)
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                // 元素交换
                Rs[i].swap(Rs[i + 1]);
                
                // 预积分交换
                // pre_integrations[i] 表示的是 从第i-1帧到第i帧的预积分
                std::swap(pre_integrations[i], pre_integrations[i + 1]);
                
                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);
                
                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            // 因为IMU正在向前推算，要保持连贯
            // 把滑动窗口的最后一帧的状态全部设置为 原本的最新帧WINDOW_SIZE（因为上面前移了，所以这里变成WINDOW_SIZE-1）
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
            
            // 将[WINDOW_SIZE-1]帧到[WINDOW_SIZE]的预积分清空
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            
            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }
                
                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
                
            }
            // 这里主要是对特征点进行处理，深度转移等
            slideWindowOld();
        }
    }
    else
    {
        // 次新帧[WINDOW_SIZE-1] 到 最新帧[WINDOW_SIZE] 的预积分，以及其他buf进行处理
        if (frame_count == WINDOW_SIZE)
        {
            // dt_buf[i]：第i-1帧到第i帧之间的imu数据的时间间隔dt集合
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                ///dt_buf[frame_count]： WINDOW_SIZE-1帧到WINDOW_SIZE帧的dt集合
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];
                
                // 将[WINDOW_SIZE-1]帧到[WINDOW_SIZE]的预积分数据,叠加到 [WINDOW_SIZE-2]到[WINDOW_SIZE-1]的预积分上
                // 因为,下一次优化的时候， 这里的 第n帧 就变成了 第n-1帧了
                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
                
                // 将[WINDOW_SIZE-1]帧到[WINDOW_SIZE]帧的dt_buf,其他buf,追加到dt_buf[WINDOW_SIZE - 1]
                // buf[frame_count - 1] : 表示 [WINDOW_SIZE-2]到[WINDOW_SIZE-1]帧之间的数据集合
                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }
            
            // 把 最新帧[WINDOW_SIZE] 状态，拷贝到[WINDOW_SIZE-1]帧
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];
            
            // 将[WINDOW_SIZE-1]帧到[WINDOW_SIZE]的预积分清空
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            
            slideWindowNew();
        }
    }
}

//滑动窗口边缘化次新帧时处理特征点被观测的帧号
//real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{

    /** ******************************************************
     * debug
     * 打印输出:
     * 1. 特征点处理之前的start_frame
     * 2. 特征点处理之后的start_frame
     * *******************************************************/
    //std::cout<<"=======================Debug START============================"<<std::endl;

    //std::cout<<"===================marg次新帧====特征点处理之前================="<<std::endl;
    //for(auto it : f_manager.feature){
    //    std::cout<<"特征点ID: ["<<it.feature_id<<"] "<<
    //               "第一帧ID: ["<<it.start_frame<<"] "<<
    //               "共视帧数: ["<<it.feature_per_frame.size()<<"] "<<std::endl;
    //}

    //==========================PROTECT: SOURCE CODE===========================
    sum_of_front++;
    f_manager.removeFront(frame_count);
    //=========================================================================

    //============================DEBUG CONTINUE===========================================
    //std::cout<<"===================marg次新帧====特征点处理之后================="<<std::endl;
    //for(auto it : f_manager.feature){
    //    std::cout<<"特征点ID: ["<<it.feature_id<<"] "<<
    //               "第一帧ID: ["<<it.start_frame<<"] "<<
    //               "共视帧数: ["<<it.feature_per_frame.size()<<"] "<<std::endl;
    //}
    //==============================DEBUG END==============================================
}

//滑动窗口边缘化最老帧时处理特征点被观测的帧号
//real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    /** ******************************************************
     * debug
     * 打印输出:
     * 1. 特征点处理之前的start_frame
     * 2. 特征点处理之后的start_frame
     * *******************************************************/

    //std::cout<<"=======================Debug START============================"<<std::endl;

    //std::cout<<"===================marg最old帧====特征点处理之前================="<<std::endl;
    //for(auto it : f_manager.feature){
    //    std::cout<<"特征点ID: ["<<it.feature_id<<"] "<<
    //               "第一帧ID: ["<<it.start_frame<<"] "<<
    //               "共视帧数: ["<<it.feature_per_frame.size()<<"] "<<std::endl;
    //}

    sum_of_back++;
    
    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        //back_R0、back_P0为窗口中最老帧的位姿
        //Rs、Ps为滑动窗口后第0帧的位姿，即原来的第1帧
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        // 进行深度转移，如果特征点的主帧被marg，那么将主帧的观测转移到主帧的下一帧(就是重新算一下深度值，将主帧改为下一帧)
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        // 不进行深度转移，也就是保持深度，但是主帧已经变成下一帧了
        f_manager.removeBack();

    //============================DEBUG CONTINUE===========================================
    //std::cout<<"===================marg最old帧====特征点处理之后================="<<std::endl;
    //for(auto it : f_manager.feature){
    //    std::cout<<"特征点ID: ["<<it.feature_id<<"] "<<
    //               "第一帧ID: ["<<it.start_frame<<"] "<<
    //               "共视帧数: ["<<it.feature_per_frame.size()<<"] "<<std::endl;
    //}
    //==============================DEBUG END==============================================
}

/**
 * @brief   进行重定位
 * @optional
 * @param[in]   _frame_stamp    重定位帧时间戳
 * @param[in]   _frame_index    重定位帧索引值
 * @param[in]   _match_points   重定位帧的所有匹配点
 * @param[in]   _relo_t     重定位帧平移向量
 * @param[in]   _relo_r     重定位帧旋转矩阵
 * @return      void
*/
void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp;
    relo_frame_index = _frame_index;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec())
        {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

