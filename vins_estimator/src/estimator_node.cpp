#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;//条件变量
double current_time = -1;
/// queue
/// “先进先出”的单向队列,只能从“后面”压进(Push)元素,从“前面”提取(Pop)元素
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;

int sum_of_wait = 0;

//互斥量
std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;

//IMU项[P,Q,B,Ba,Bg,a,g]
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

//从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，中值积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();

    //init_imu=1表示第一个IMU数据
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }

    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    // 取上一个时刻的的加速度（世界坐标系下的）
    // tmp_Q:最新的姿态（机体坐标系到世界坐标系的旋转）
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    // 中值法积分
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // 更新当前时刻的姿态预测
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    // 推算当前时刻的加速度（世界坐标系下）
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    // 中值法，得到加速度
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 位置推算
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    // 速度推算
    tmp_V = tmp_V + dt * un_acc;

    // 记录线速度、角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
//对imu_buf中剩余的imu_msg进行PVQ递推
void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

//取 <前一帧图像与当前帧图像数据之间的IMU数据，当前帧图像的特征点数据>
/**
 * @brief   对imu和图像数据进行对齐并组合
 * @Description     img:    i -------- j  -  -------- k
 *                  imu:    - jjjjjjjj - j/k kkkkkkkk -
 *                  直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据
 * @return  vector<std::pair<vector<ImuConstPtr>, PointCloudConstPtr>> (IMUs, img_msg)s
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        // imu数据队列为空 或者 特征点数据队列为空
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;    //数据取完了，返回数据

        // 条件：IMU最后一个数据的时间要 > 第一个图像特征数据的时间
        // 如果：最新imu数据时间戳 < 最旧图像数据时间戳 ======> (IMU数据落后了，太慢了，等IMU)
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;  //(等待iMU的次数统计)
            return measurements;    // 先把已有的数据返回了先
        }

        // 条件：IMU第一个数据的时间要 < 第一个图像特征数据的时间
        // 如果：最旧IMU数据时间戳 > 最新图像数据时间戳 ======> 图像数据落后了，直接把该图像数据丢弃
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            // pop掉 队列最前面的数据，然后继续\while
            feature_buf.pop();
            continue;
        }

        // 取图像（特征点）数据队列第一个数据
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        // 弹出图像数据队列第一个数据
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        // 循环: 如果（imu队列最前面[最旧]元素时间戳 < 刚刚弹出来的图像队列第一个数据的时间戳+等待时间）
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            // 开始截取 前一帧图像i与新一帧图像j数据之间的 IMU数据
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        //这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());

        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

//imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();

    con.notify_one();//唤醒作用于process线程中的获取观测值数据的函数

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        //构造互斥锁m_state，析构时解锁
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);//递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        //发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

//feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();

    con.notify_one();
}

//restart回调函数，收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");

        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();

        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();

        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

//relocalization回调函数，将points_msg放入relo_buf
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
/**
 * @brief   VIO的主线程
 * @Description 等待并获取measurements：(IMUs, img_msg)s，计算dt
 *              estimator.processIMU()进行IMU预积分
 *              estimator.setReloFrame()设置重定位帧
 *              estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化
 * @return      void
*/
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);

        //条件变量允许我们通过通知进而实现线程同步。
        //因此，您可以实现发送方/接收方或生产者/消费者之类的工作流。
        //在这样的工作流程中，接收者正在等待发送者的通知。如果接收者收到通知，它将继续工作

        //当 std::condition_variable 对象的某个 wait 函数被调用的时候，
        //它使用 std::unique_lock(通过 std::mutex) 来锁住当前线程。
        //当前线程会一直被阻塞，直到另外一个线程在相同的 std::condition_variable 对象上调用了
        //notification 函数来唤醒当前线程
        con.wait(lk, [&]
        {
            // 这里的return ， 只是 这个{}括号内函数return，不是void process()的return
            // getMeasurements(): 取 <前一帧图像与当前帧图像数据之间的IMU数据，当前帧图像的特征点数据>
            return (measurements = getMeasurements()).size() != 0;
        });
        lk.unlock();
        m_estimator.lock();
        // 开始处理返回的数据
        for (auto &measurement : measurements)
        {
            // 取图像数据
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            // 遍历imu数据（imu数据是第i帧图像到第j帧图像之间的）
            for (auto &imu_msg : measurement.first)
            {
                // 取imu数据时间戳
                double t = imu_msg->header.stamp.toSec();
                // 取第j帧图像数据时间戳+td
                double img_t = img_msg->header.stamp.toSec() + estimator.td;

                // imu数据时间 < 第j帧图像数据时间戳
                if (t <= img_t)
                {
                    // 取最早的时间作为 `current_time`
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;       // dt = imu时间-系统时间
                    ROS_ASSERT(dt >= 0);                // 检查系统时间是否 < imu时间
                    current_time = t;                   // 系统时间 = imu时间

                    // 直接使用imu数据
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // IMU预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    // imu数据时间 > 第j帧图像数据时间戳
                    // (就是跟第j帧时间戳靠的很近的iMU数据，应该需要计算预积分的最后一份imu数据了，这个imu数据在图像时间戳的后面)
                    // 需要插值，把IMU数据插值到 刚好是 第j帧图像数据时间戳
                    double dt_1 = img_t - current_time;     // dt = 第j帧图像数据时间戳 - 系统时间
                    double dt_2 = t - img_t;                // dt2 = imu 数据时间 -  第j帧图像数据时间戳
                    current_time = img_t;                   // 系统时间设置为 ： 第j帧图像数据时间 （预积分的终点）
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    // 线性插值
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    // 执行最后一次积分，执行完之后，刚好积分到 第j帧图像数据时间
                    // 完成从第i帧图像到第j帧图像之间的预积分计算
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;

            //取出最后一个重定位帧
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }

            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;

            // 将每个特征点的数据打包成<特征点id，(camera_id,[x,y,z,u,v,vx,vy])>
            // 存到image容器,该容器以[特征点id]为key，来检索元素
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                // 特征点的归一化平面点
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                // 特征点像素坐标
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                // 归一化平面点速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                // 打包，存到image容器
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }

            //==========================Debug START=====================================

            /** ******************************************************
             * debug
             * 根据IMU积分得到的位姿，作为初始值，
             * 将上一帧的路标点投影到当前帧的像素坐标系下，检查与LK跟踪的差异
             * *******************************************************/

//            // 取当前帧由IMU积分得到的机体位姿
//            int frame_count_=estimator.frame_count;
//            Matrix3d Rwj=estimator.Rs[frame_count_];     //姿态（机体坐标系到世界坐标系的旋转）
//            Vector3d Pwj=estimator.Ps[frame_count_];

//            map<int,Vector3d> landMark_;
//            // 取所有路标点
//            for (auto &it_per_id : estimator.f_manager.feature)
//            {
//                int used_num;
//                used_num = it_per_id.feature_per_frame.size();
//                if (!(used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
//                    continue;
//                if (it_per_id.start_frame > WINDOW_SIZE * 3.0 / 4.0 || it_per_id.solve_flag != 1)
//                    continue;
//                int imu_i = it_per_id.start_frame;
//                // 取路标点数据
//                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
//                // 转换为世界坐标系的3D点
//                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];
//                landMark_[it_per_id.feature_id]=w_pts_i;
//            }

//            // 遍历当前帧的特征点
//            map<int,pair<cv::Point2f,cv::Point2f>> point_pair_2_show;
//            for (unsigned int i = 0; i < img_msg->points.size(); i++)
//            {
//                int v = img_msg->channels[0].values[i] + 0.5;
//                int feature_id = v / NUM_OF_CAM;
//                // 特征点的归一化平面点
//                double nor_x_lk = img_msg->points[i].x;
//                double nor_y_lk = img_msg->points[i].y;
//                double nor_z_lk = img_msg->points[i].z;

//                auto it = landMark_.find(feature_id);
//                if(it != landMark_.end()){
//                    // 滑动窗口中的路标点转换到当前帧, 世界坐标系-->当前帧机体坐标系-->相机坐标系
//                    Vector3d pt_in_j = Rwj.transpose()*it->second - Rwj.transpose()*Pwj;
//                    pt_in_j= estimator.ric[0].transpose() * pt_in_j -
//                            estimator.ric[0].transpose()*estimator.tic[0];

//                    // 深度值为0的去掉
//                    if(pt_in_j.z()<=0)
//                        continue;

//                    // 得到归一化平面点，转像素坐标系
//                    // 这里直接用 给定的标准相机模型好了， 真正的相机模型不好取
//                    // IMU积分位姿得到的点
//                    double x_imu = FOCAL_LENGTH * pt_in_j.x() / pt_in_j.z() + COL / 2.0;
//                    double y_imu = FOCAL_LENGTH * pt_in_j.y() / pt_in_j.z() + ROW / 2.0;
//                    // LK光流跟踪得到的点
//                    double x_lk = FOCAL_LENGTH * nor_x_lk / nor_z_lk + COL / 2.0;
//                    double y_lk = FOCAL_LENGTH * nor_y_lk / nor_z_lk + ROW / 2.0;

//                    if((x_imu < 0 || x_imu > COL) || (x_lk <0 || x_lk > COL) ||
//                       (y_imu < 0 || y_imu > ROW) || (y_lk <0 || y_lk > ROW))
//                        continue;

//                    point_pair_2_show[feature_id] = pair<cv::Point2f,cv::Point2f>
//                            (cv::Point2f(x_imu, y_imu),cv::Point2f(x_lk, y_lk));
//                }
//            }

//            // 显示
//            cv::Mat img2show(ROW,COL,CV_8UC3,cv::Scalar(0, 0,0));
//            for (auto &it : point_pair_2_show){
//                cv::circle(img2show,cv::Point2f(it.second.first),3,cv::Scalar(0,0,255),1);
//                cv::circle(img2show,cv::Point2f(it.second.second),3,cv::Scalar(0,255,0),1);
//            }
//            cv::imshow("show the point",img2show);
//            cv::waitKey(10);

            //==========================Debug END=====================================

            //处理图像特征
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            //给RVIZ发送topic
            pubOdometry(estimator, header);//"odometry" 最新帧IMU位姿信息PQV
            pubKeyPoses(estimator, header);//"key_poses" 滑动窗口三维坐标
            pubCameraPose(estimator, header);//"camera_pose"  前一帧相机位姿
            pubPointCloud(estimator, header);//"history_cloud" 点云信息
            pubTF(estimator, header);//"extrinsic" 相机到IMU的外参，以及body到世界坐标系的TF变换
            pubKeyframe(estimator);//"keyframe_point"、"keyframe_pose" 关键帧位姿和点云

            if (relo_msg != NULL)
                pubRelocalization(estimator);//"relo_relative_pose" 重定位位姿
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();//更新IMU参数[P,Q,V,ba,bg,a,g]
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    //ROS初始化，设置句柄n
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    //读取参数，设置估计器参数
    readParameters(n);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // 注册消息发布器
    registerPub(n);

    // 订阅IMU
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    // 订阅前端跟踪topic，绑定回调
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // 订阅重启topic，绑定回调
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // 订阅重定位topic，绑定回调
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    //创建VIO主线程
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
