#include "feature_tracker.h"

//FeatureTracker的static成员变量n_id初始化为0
int FeatureTracker::n_id = 0;

//判断跟踪的特征点是否在图像边界内
bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    //cvRound()：返回跟参数最接近的整数值，即四舍五入；
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

//去除无法跟踪的特征点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    // 遍历v
    for (int i = 0; i < int(v.size()); i++)
        // 去掉v容器里面，值为0的元素，重新排序
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//去除无法追踪到的特征点
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

//空的构造函数
FeatureTracker::FeatureTracker()
{
}

/**
 * @brief   对跟踪点进行排序并去除密集点
 * @Description 对跟踪到的特征点，按照被追踪到的次数排序并依次选点
 *              使用mask进行类似非极大抑制，半径为30，去掉密集点，使特征点分布均匀
 * @return      void
*/
void FeatureTracker::setMask()
{
    //鱼眼跟DSO那个vitn相片一样，只要拿有效区域
    //这个mask是为了避免提角点都提到一块了用的。如果能识别动态物体，也可以把识别到的物体那块也设置mask
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    // 构造(cnt，pts，id)序列
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    // 储存特征点到cnt_pts_id容器，<被观测数,<特征点像素坐标，特征点唯一id>>
    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    //对光流跟踪到的特征点forw_pts，按照被跟踪到的次数cnt从大到小排序（lambda表达式）
    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    // 根据mask，重新挑选当前帧的特征点
    //清空cnt，pts，id并重新存入
    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    // 遍历正在跟踪的特征点
    for (auto &it : cnt_pts_id)
    {
        // 如果mask在该特征点处的值为255
        if (mask.at<uchar>(it.second.first) == 255)
        {
            //当前特征点位置对应的mask值为255，则保留当前特征点，将对应的特征点位置pts，id，被追踪次数cnt分别存入
            forw_pts.push_back(it.second.first);    // 记录当前帧的特征点像素坐标
            ids.push_back(it.second.second);        // 记录正在跟踪的特征点的id
            track_cnt.push_back(it.first);          // 记录特征点被观测次数
            // 在该特征点附近画一个圈，表示在这里附近就不再提取特征点了
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

// 当前帧特征点添加这些新检测的点
void FeatureTracker::addPoints()
{
    // 遍历新检测的特征点n_pts
    for (auto &p : n_pts)
    {
        // 当前帧特征点添加这些新检测的点
        forw_pts.push_back(p);
        // 扩充一下 ids 容器
        ids.push_back(-1);//新提取的特征点id初始化为-1
        // 这些新检测的点被观测次数=1
        track_cnt.push_back(1);
    }
}

/**
 * @brief   对图像使用光流法进行特征点跟踪
 * @Description createCLAHE() 对图像进行自适应直方图均衡化
 *              calcOpticalFlowPyrLK() LK金字塔光流法
 *              setMask() 对跟踪点进行排序，设置mask
 *              rejectWithF() 通过基本矩阵剔除outliers
 *              goodFeaturesToTrack() 添加特征点(shi-tomasi角点)，确保每帧都有足够的特征点
 *              addPoints()添加新的追踪点
 *              undistortedPoints() 对角点图像坐标去畸变矫正，并计算每个角点的速度
 * @param[in]   _img 输入图像
 * @param[in]   _cur_time 当前时间（图像时间戳）
 * @return      void
*/
void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;   //记录图像的时间戳

    //如果EQUALIZE=1，表示太亮或太暗，进行直方图均衡化处理
    if (EQUALIZE)
    {
        //自适应直方图均衡
        //createCLAHE(double clipLimit, Size tileGridSize)
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    // 检查前一帧是否为空
    if (forw_img.empty())
    {
        //如果当前帧的图像数据forw_img为空，说明当前是第一次读入图像数据
        //将读入的图像赋给当前帧forw_img，同时还赋给prev_img、cur_img
        //储存当前帧到前前一帧、前一帧，以及当前帧
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        //否则，说明之前就已经有图像读入，只需要更新当前帧forw_img的数据
        forw_img = img;
    }

    //此时forw_pts还保存的是上一帧图像中的特征点，所以把它清除
    forw_pts.clear();

    //如果前一帧的特征点数>0
    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        // 光流法，跟踪
        // (前一帧，当前帧，前一帧的点，当前帧的点，跟踪状态，误差，窗口大小，金字塔层数)
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        // 通过跟踪状态、以及是否在边界内，设置status
        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        // 根据status容器，去掉前面容器里面，status对应值为0的元素，重新排序
        //不仅要从当前帧数据forw_pts中剔除，而且还要从cur_un_pts、prev_pts和cur_pts中剔除
        //prev_pts和cur_pts中的特征点是一一对应的
        //记录特征点id的ids，和记录特征点被跟踪次数的track_cnt也要剔除
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    // 当前跟踪的点的计数
    //光流追踪成功,特征点被成功跟踪的次数就加1
    //数值代表被追踪的次数，数值越大，说明被追踪的就越久
    for (auto &n : track_cnt)
        n++;

    //PUB_THIS_FRAME=1 需要发布特征点
    if (PUB_THIS_FRAME)
    {
        // 剔除一些跟踪错误的特征点
        rejectWithF();
        ROS_DEBUG("set mask begins");
        TicToc t_m;

        setMask();//保证相邻的特征点之间要相隔30个像素,设置mask
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        // 如果当前跟踪的特征点还没达到上限，则再提取一些特征点
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;

            /// 提取新的特征点
            /// mask: 避免在老点附近提取
            /**
             *void cv::goodFeaturesToTrack(    在mask中不为0的区域检测新的特征点
             *   InputArray  image,              输入图像
             *   OutputArray     corners,        存放检测到的角点的vector
             *   int     maxCorners,             返回的角点的数量的最大值
             *   double  qualityLevel,           角点质量水平的最低阈值（范围为0到1，质量最高角点的水平为1），小于该阈值的角点被拒绝
             *   double  minDistance,            返回角点之间欧式距离的最小值
             *   InputArray  mask = noArray(),   和输入图像具有相同大小，类型必须为CV_8UC1,用来描述图像中感兴趣的区域，只在感兴趣区域中检测角点
             *   int     blockSize = 3,          计算协方差矩阵时的窗口大小
             *   bool    useHarrisDetector = false,  指示是否使用Harris角点检测，如不指定则使用shi-tomasi算法
             *   double  k = 0.04                Harris角点检测需要的k值
             *)
             */
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        // 将新检测的特征点添加到当前帧特征点容器
        //添将新检测到的特征点n_pts添加到forw_pts中，id初始化-1,track_cnt初始化为1.
        addPoints();
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }

    /** ******************************************************************
     * Debug: QPC
     * func: 打印输出特征点id以及被跟踪次数
     ********************************************************************/

//    std::cout<<"================Debug START==================="<<std::endl;
//    // 对ID和被跟踪次数，进行打包pair<>
//    vector<pair<int,int>> pair_id_cnts;
//    if(ids.size()>0){
//        for(int i=0;i<ids.size();i++){
//            pair_id_cnts.emplace_back(pair<int,int>(ids[i],track_cnt[i]));
//            //std::cout<<"Feature Point [ID]: "<<ids_tmp[i]<<"    Tracked Count: "<<track_cnt[i]<<std::endl;
//        }
//    }
//    // 对pair<> 按ID号排序，方便查看
//    sort(pair_id_cnts.begin(), pair_id_cnts.end(), [](const pair<int,int> &a, const pair<int,int> &b)
//         {
//            return a.first > b.first;
//         });
//    // 打印输出
//    if(pair_id_cnts.size()>0){
//        for(auto &element : pair_id_cnts)
//        std::cout<<"Feature Point [ID]: "<<element.first<<"    Tracked Count: "<<element.second<<std::endl;
//    }
//    std::cout<<"================Debug  END==================="<<std::endl;

    // 储存一下
    prev_img = cur_img;         //上上一帧
    prev_pts = cur_pts;         //上上一帧特征点
    prev_un_pts = cur_un_pts;   //上上一帧去畸变的归一化平面点
    cur_img = forw_img;         //当前帧赋值给上一帧
    cur_pts = forw_pts;         //当前帧特征点赋值给上一帧特征点
    undistortedPoints();        //根据不同的相机模型去畸变矫正和转换到归一化坐标系上，计算速度
    prev_time = cur_time;       //记录时间戳
}

/**
 * @brief   通过F矩阵去除outliers
 * @Description 将图像坐标转换为归一化坐标
 *              cv::findFundamentalMat()计算F矩阵
 *              reduceVector()去除outliers
 * @return      void
*/
void FeatureTracker::rejectWithF()
{
    // 如果LK光流法跟踪到的点数>8，开始筛选
    if (forw_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;

        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {

            Eigen::Vector3d tmp_p;
            // 输入: 畸变的像素点（这个畸变的像素点由畸变的归一化平面点产生），
            //          (同时，这个畸变的像素点其实对应这真实世界坐标系的某个点，
            //          但是由于透镜畸变，使得直接使用相机模型反投影得到的归一化平面点是畸变的，与真实世界不是成比例的)
            // 得到: 去畸变之后 的归一化平面点 tmp_p, 这个去畸变的归一化平面点才能与真实世界成比例
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            // 使用另外的标准相机模型，将无畸变的归一化平面点转换到像素平面
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        // 根据两帧的无畸变的匹配像素点，求解两帧的F矩阵，同时筛选点
        vector<uchar> status;
        //调用cv::findFundamentalMat对un_cur_pts和un_forw_pts计算F矩阵
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        // 根据筛选状态，在各个容器中去除一些被剔除的点，resize容器
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

// i会从0开始，重复调用，
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        // ids[i] == -1，表示是新添加的特征点，还没有光流跟踪
        if (ids[i] == -1)
            ids[i] = n_id++;    // 记录为这个点的id，然后总点数++
        return true;
    }
    else
        return false;
}

//读取相机内参
void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    // 根据相机内参，生成一个相机对象（工厂模式）
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}


//显示去畸变矫正后的特征点  name为图像帧名称
void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}


//对角点图像坐标进行去畸变矫正，转换到归一化坐标系上，并计算每个角点的速度。
void FeatureTracker::undistortedPoints()
{
    // 清空： 当前帧 去畸变归一化平面点容器
    cur_un_pts.clear();
    // 清空： 当前帧 特征点数据容器
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());

    // 遍历当前帧特征点(因为在前一步，已经赋值给cur_pts了)
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        // 取一个点
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        // 输入: 像素点
        // 得到: 去畸变之后 的归一化平面点 b, 这个去畸变的归一化平面点才能与真实世界成比例
        m_camera->liftProjective(a, b);
        // 保存归一化平面点
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));

        ///cur_un_pts_map: 记录当前帧正在跟踪的特征点数据 <特征点ID（唯一的），特征点在当前帧的归一化平面坐标>
        // ids[i]=-1，表示是新添加的点，还没有进行光流跟踪
        // ids[i]: 当前这个特征点的ID，是唯一的
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }

    /// caculate points velocity
    // 计算每个特征点的速度到pts_velocity
    if (!prev_un_pts_map.empty())
    {
        // dt
        double dt = cur_time - prev_time;
        pts_velocity.clear();

        // 遍历当前帧的归一化平面点
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            // ids[i]=-1，表示是新添加的点，还没有进行光流跟踪
            if (ids[i] != -1)
            {
                // 这里只对进行过光流跟踪的点进行操作
                std::map<int, cv::Point2f>::iterator it;

                // 根据(正在遍历的)当前帧特征点i的唯一ID，在上一帧中寻找对应的像素
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())    //找到，则计算两帧之间，特征点的移动速度，归一化平面上
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
