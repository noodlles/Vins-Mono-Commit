#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

//清楚特征管理器中的特征
void FeatureManager::clearState()
{
    feature.clear();
}

//窗口中被跟踪的有效特征数量
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    // 遍历feature容器
    for (auto &it : feature)
    {
        // 取每一个特征点被跟踪的次数
        it.used_num = it.feature_per_frame.size();

        // 被跟踪次数满足一定条件
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            // 把该特征点作为有效特征点，计数+1
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief   把特征点放入feature的list容器中，计算每一个点跟踪次数和它在次新帧和次次新帧间的视差，返回是否是关键帧
 * @param[in]   frame_count 窗口内帧的个数
 * @param[in]   image 某帧所有特征点的[camera_id,[x,y,z,u,v,vx,vy]]s构成的map,索引为feature_id
 * @param[in]   td IMU和cam同步时间差
 * @return  bool true：次新帧是关键帧;false：非关键帧
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    // 遍历来自前端的特征点数据
    // 把image map中的所有特征点放入`feature`这个list容器中
    for (auto &id_pts : image)
    {
        // id_pts : <特征点id，(camera_id,[x,y,z,u,v,vx,vy])>
        // id_pts.second[0].second: 取[x,y,z,u,v,vx,vy]这部分
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        // 迭代器寻找feature容器中是否已经有这个特征点
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 如果没有则新建一个FeaturePerId对象，并添加这图像帧
        if (it == feature.end())
        {
            // 新建一个FeaturePerId对象，push到feature
            feature.push_back(FeaturePerId(feature_id, frame_count));
            // 对新建的FeaturePerId对象进行操作，操作其成员变量`feature_per_frame`容器
            // 把观测到该特征点的该帧的数据构成FeaturePerFrame对象，然后保存到FeaturePerId对象的`feature_per_frame`容器中
            // FeaturePerFrame对象数据包括： （[x,y,z,u,v,vx,vy]，时间戳？)
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        // 如果已经有对应的FeaturePerId对象，把新的观测信息添加到FeaturePerId对象的`feature_per_frame`容器中
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++;
        }
    }

    // 如果总帧数<2，才启动， 或者上一次跟踪的点数<20
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 遍历feature容器
    // 计算每个特征在次新帧和次次新帧中的视差
    for (auto &it_per_id : feature)
    {
        // 如果遍历到的特征点
        // 第一次观测到它的帧号 <= 当前帧号-2 ，表示距离第一次观测已经超过2帧了
        // 同时，
        //
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1)
        {
            // 利用观测到该特征点的上上一帧和上一帧的数据，计算视差
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

// 获取frame_count_l与frame_count_r两帧之间的对应(关联)特征点
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    // 遍历特征点
    for (auto &it : feature)
    {
        // 如果特征点的起始帧 < frame_l  &&  特征点最后被观测的帧 >= frame_r ====> 表示特征点在两帧都有被观测到
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            // 取特征点在 frame_l 的归一化平面点
            a = it.feature_per_frame[idx_l].point;

            // 取特征点在 frame_r 的归一化平面点
            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

//设置特征点的逆深度估计值
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 逆深度重新转换为深度值
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;//失败估计
        }
        else
            it_per_id.solve_flag = 1;//成功估计
    }
}

//剔除feature中估计失败的点（solve_flag == 2）
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    // 遍历特征点
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        // 设置点的逆深度
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

// 获取特征点逆深度
VectorXd FeatureManager::getDepthVector()
{
    // 窗口中被跟踪的有效特征数量
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        // 遍历特征点，根据特征点被观测次数，检查是否满足，不满足则跳过，不参与优化
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        /// 这里使用的是逆深度
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

// 对滑动窗口中的特征点进行三角化求深度（SVD分解）
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    // tic[]: imu和相机的平移外参,从相机到imu的平移变换
    // ric[]: 旋转外参

    // 遍历feature容器
    for (auto &it_per_id : feature)
    {
        // 某个特征点it_per_id

        // 取被观测次数
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 如果已经有深度值，则跳过
        if (it_per_id.estimated_depth > 0)
            continue;
        // 取第一次观测到该特帧点ID
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);

        // 准备矩阵A [rows: 2*观测次数, cols: 4]
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        //R0 t0为第i帧相机坐标系到世界坐标系的变换矩阵
        Eigen::Matrix<double, 3, 4> P0;
        // Ps[imu_i]: 第i帧机体坐标系在世界坐标系的坐标
        // Rs[imu_i]: 机体坐标系到世界坐标系的旋转
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];    //t0: 相机坐标系在世界坐标系的平移
        // ric[0]： 相机坐标系到iMU坐标系的旋转变换
        // Rs[imu_i]: IMU坐标系到世界坐标系的旋转变换
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];

        // 投影矩阵
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        // 遍历观测到该特征点的每一个图像帧数据
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            //R t为第j帧相机坐标系到第i帧相机坐标系的变换矩阵，P为i到j的变换矩阵
            //第j帧图像时刻
            //t1: 第j帧相机坐标系到世界坐标系的平移
            //R1: 第j帧相机坐标系到世界坐标系的旋转
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];

            // 这里可以使用T_w1^{-1}*T_w2 展开，就明显了
            // t: 第j帧相机坐标系到第i帧相机坐标系的平移变换
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            // R: 第j帧相机坐标系到第i帧相机坐标系的旋转变换
            Eigen::Matrix3d R = R0.transpose() * R1;

            // 构造P矩阵
            Eigen::Matrix<double, 3, 4> P;
            // 重新设置成 ： 第i帧相机坐标系到第j帧相机坐标系的变换
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            // 归一化平面点
            Eigen::Vector3d f = it_per_frame.point.normalized();

            // 构造方程组
            //P = [P1 P2 P3]^T
            //AX=0      A = [A(2*i) A(2*i+1) A(2*i+2) A(2*i+3) ...]^T
            //A(2*i)   = x(i) * P3 - z(i) * P1
            //A(2*i+1) = y(i) * P3 - z(i) * P2

            //qpc: 这里的f[2]应该都等于1吧
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        // 求出了最小奇异值对应的奇异值向量之后，需要对第4维进行一个归一化，才能真正的三角化
        // 对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        // 只保留z值(深度值)即可
        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        // 如果深度值很小，直接设置为固定值
        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

//移除外点
void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

//边缘化最老帧时，处理特征点保存的帧号，将起始帧是最老帧的特征点的深度值进行转移
//marg_R、marg_P为被边缘化的位姿，new_R、new_P为在这下一帧的位姿
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    // 遍历特征点
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        //特征点起始帧(主帧)不是最老帧则将帧号减一
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // 特征点起始帧是最老帧
            // 取特征点在主帧上的归一化平面坐标
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            // 删除主帧
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            // 特征点只在最老帧被观测则直接移除整个特征点，不需要转移了
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                ///将特征点的主帧转移给下一帧
                // pts_i为特征点在最老帧坐标系下的三维坐标
                // w_pts_i为特征点在世界坐标系下的三维坐标
                // 将其转换到在下一帧坐标系下的坐标pts_j
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

//边缘化最老帧时，直接将特征点所保存的帧号向前滑动
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 如果特征点的主帧不是要被marg的帧，则其主帧号(start_frame)减一，因为最odl的帧被marg了
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // 如果特征点的主帧是要被marg的帧，那么，直接删除主帧关于这个特征点的观测（不进行深度转移）
            // 如果feature_per_frame（可以观测到这个特征点的帧）为空，则直接删除特征点
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//边缘化次新帧时，对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        // 如果特征点的起始帧为最新帧[WINDOW_SIZE]，
        // 那么，在下一次优化的时候，特征点的起始帧为[WINDOW_SIZE-1]帧
        // 将该特征点的主帧(start_frame) 变成次新帧-1
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            // 否则，特征点被其他更早的帧观测到

            int j = WINDOW_SIZE - 1 - it->start_frame;
            // 如果次新帧之前,这个特征点已经失去跟踪，则跳过
            // endFrame： 最后跟踪到该特征点的帧
            if (it->endFrame() < frame_count - 1)
                continue;
            // 否则，表示该特征点在次新帧仍被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            // 如果该特征点只被次新帧跟踪到
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

//计算某个特征点it_per_id在次新帧和次次新帧的视差
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    // 检查上一帧是否为关键帧
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    // 观测到该点的上上一帧
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    // 观测到该点的上一帧
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    // 归一化平面坐标
    Vector3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    // 归一化平面坐标
    Vector3d p_i = frame_i.point;
    Vector3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;

    p_i_comp = p_i;

    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    // 计算差值
    double du = u_i - u_j, dv = v_i - v_j;

    // 再归一化一下
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    // 计算差值
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 取非负的
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}
