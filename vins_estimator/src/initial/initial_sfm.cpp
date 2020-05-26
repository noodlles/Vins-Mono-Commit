#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

/**
 * @brief GlobalSFM::triangulatePoint
 *        三角化两帧间某个对应特征点的深度
 * @param Pose0     参考坐标系到frame0坐标系的变换
 * @param Pose1     参考坐标系到frame1坐标系的变换
 * @param point0    特征点在frame0的归一化平面坐标
 * @param point1    特征点在frame1的归一化平面坐标
 * @param point_3d  三角化结果[输出]
 */
void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * @brief GlobalSFM::solveFrameByPnP
 *        PNP方法得到第l帧到第i帧的R_initial、P_initial
 * @param R_initial     从参考帧到该帧的变换
 * @param P_initial     从参考帧到该帧的变换
 * @param i             该帧id (第i帧)
 * @param sfm_f         滑窗所有特征点
 * @return
 */
bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
    // 遍历特征点
	for (int j = 0; j < feature_num; j++)
	{
        // 如果还没有三角化，则跳过该特征点
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
        // 遍历该特征点的所有观测
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
            // 如果该特征点起始帧为第i帧
			if (sfm_f[j].observation[k].first == i)
			{
                // 取观测值(在该帧的归一化平面坐标)
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
                // 取该特征点在参考坐标系的3D坐标
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
    // 检查点数是否足够
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
    // 从参考帧坐标系到第i帧的变换
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    // 调用cv::solvePnP
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
    // 罗德里格斯公式
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

/**
 * @brief GlobalSFM::triangulateTwoFrames
 *          三角化frame0和frame1间所有对应点
 * @param frame0    frame0
 * @param Pose0     参考坐标系到frame0坐标系的变换
 * @param frame1    frame1
 * @param Pose1     参考坐标系到frame1坐标系的变换
 * @param sfm_f     滑动窗口所有特征点
 */
void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
    // 遍历特征点
	for (int j = 0; j < feature_num; j++)
	{
        // 如果已经三角化过，则跳过
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
        // 遍历该特征点的所有观测数据
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
            /// k为特征点被观测顺序
            // 如果frame0观测到该特征点
			if (sfm_f[j].observation[k].first == frame0)
			{
                // 记录观测数据
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
            // 如果frame1观测到该特征点
			if (sfm_f[j].observation[k].first == frame1)
			{
                // 记录观测数据
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
        // 两帧都能观测到特征点
		if (has_0 && has_1)
		{
            // 三角化
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
            // 保存三角化结果
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

/**
 * @brief   纯视觉sfm，求解窗口中的所有图像帧的位姿和特征点坐标
 * @param[in]   frame_num	窗口总帧数（frame_count + 1）
 * @param[out]  q 	窗口内图像帧的旋转四元数q（相对于参考帧[第l帧]，到第l帧的旋转变换）
 * @param[out]	T 	窗口内图像帧的平移向量T（相对于参考帧[第l帧]，到第l帧的平移变换）
 * @param[in]  	l 	第l帧
 * @param[in]  	relative_R	当前帧到第l帧的旋转矩阵
 * @param[in]  	relative_T 	当前帧到第l帧的平移向量
 * @param[in]  	sfm_f		所有特征点
 * @param[out]  sfm_tracked_points 所有在sfm中三角化的特征点ID和坐标
 * @return  bool true:sfm求解成功
*/
// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view

    // 第l帧作为参考帧
    // 假设第l帧为原点，根据当前帧到第l帧的relative_R，relative_T，得到当前帧位姿
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
    // 设置最新帧相对于参考帧的位姿 （最新帧到第l帧的变换）
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

    // 这里的位姿c_Rotation, c_Translation 这些，都是表示 从参考帧[第l帧]到滑动窗口每一帧的变换
	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];
	//这里的pose表示的是第l帧到每一帧的变换矩阵

    // 逆一下
	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    // 逆一下，因为q[],c_Quat[],T[] 都是表示 从参考帧到其他帧的变换
	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];

    // 1: trangulate between l ----- frame_num - 1
    // 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
    /// 1、先三角化第l帧（参考帧）与第frame_num-1帧（当前帧）的路标点
    /// 2、pnp求解[从第l+1开始到frame_num-1帧（当前帧）]的位姿R_initial, P_initial，保存在Pose中
    /// 2.2 对 "[从第l+1帧开始的每一帧]" 与 当前帧 进行三角化
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
            // 取前一帧在参考坐标系的位姿作为初始值(从参考帧到该帧的变换)
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
            // pnp求解: 从参考帧坐标系到第i帧的变换
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
            // 记录结果
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

        // sfm_f： 滑动窗口所有特征点
		// triangulate point based on the solve pnp result
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
    //   [frame_num -2]: 也就是当前帧的前一帧
    ///3、对 第l帧[参考帧] 与 从 "第l+1帧到当前帧" 的每一帧 进行三角化
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);

	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
    ///4、PNP求解 [从"第l-1到第0帧"的每一帧] 与 [第l帧] 之间的变换矩阵，并进行三角化
	for (int i = l - 1; i >= 0; i--)
	{
        // solve pnp
        // 取后一帧在参考坐标系的位姿作为初始值(从参考帧到该帧的变换)
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];

        // triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f);
	}

	//5: triangulate all other points
    //5、三角化其他未恢复的特征点。
    //至此得到了滑动窗口中所有图像帧的位姿以及特征点的3d坐标
	for (int j = 0; j < feature_num; j++)
	{
        // 该特征点已经三角化完成，则跳过
		if (sfm_f[j].state == true)
			continue;
        // 否则，如果该特征点被观测次数>=2，那么可以尝试进行三角化
		if ((int)sfm_f[j].observation.size() >= 2)
		{
            // 取该特征点的起始帧的观测
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first;
			point0 = sfm_f[j].observation[0].second;
            // 取最后观测到该特征点的那一帧观测数据
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
            // 三角化
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
    //6、使用cares进行全局BA优化
	ceres::Problem problem;
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
	//cout << " begin full BA " << endl;
    // 遍历每一帧
	for (int i = 0; i < frame_num; i++)
	{
		//double array for ceres
        // 设置参数块: 待优化变量
        c_translation[i][0] = c_Translation[i].x();     //平移量
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quat[i].w();               //旋转量
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
		problem.AddParameterBlock(c_translation[i], 3);
        // 固定参考帧的旋转量
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_rotation[i]);
		}
        // 固定参考帧和当前帧的平移量
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]);
		}
	}

    // 遍历滑动窗口中的特征点
	for (int i = 0; i < feature_num; i++)
	{
        // 如果还没有三角化，则跳过该点
		if (sfm_f[i].state != true)
			continue;
        // 遍历该特征点的所有观测
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
            // 取第一次观测到该特征点的归一化平面坐标
			int l = sfm_f[i].observation[j].first;
            //
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y());

            // 与ReprojectionError3D::operator() 的顺序保持一致
            //
            problem.AddResidualBlock(cost_function,         // 上面构造的csot func
                                     NULL,                  // 核函数
                                     c_rotation[l],         // 待优化旋转量
                                     c_translation[l],      // 待优化平移量
                                     sfm_f[i].position);    // 待优化点的3D坐标
		}

	}
    // 使用SCHUR求解
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	//std::cout << summary.BriefReport() << "\n";
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}

	//这里得到的是第l帧坐标系到各帧的变换矩阵，应将其转变为各帧在第l帧坐标系上的位姿
	for (int i = 0; i < frame_num; i++)
	{
		q[i].w() = c_rotation[i][0]; 
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse();
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
    // 根据优化值，更新
	for (int i = 0; i < frame_num; i++)
	{
        // 平移量更新
		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
    // 更新路标点坐标
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

