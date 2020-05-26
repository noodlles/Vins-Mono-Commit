#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "utility/tic_toc.h"
#include "utility/utility.h"
#include "parameters.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

using namespace Eigen;
using namespace std;
using namespace DVision;

/**
* @class BriefExtractor
* @Description 通过Brief模板文件，对图像的关键点计算Brief描述子
*/
class BriefExtractor
{
public:
  virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
  BriefExtractor(const std::string &pattern_file);

  DVision::BRIEF m_brief;
};

/**
* @class KeyFrame
* @Description 构建关键帧，通过BRIEF描述子匹配关键帧和回环候选帧
*/
class KeyFrame
{
public:
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
			 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal, 
			 vector<double> &_point_id, int _sequence);
	KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, Vector3d &_T_w_i, Matrix3d &_R_w_i,
			 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
			 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<BRIEF::bitset> &_brief_descriptors);
	bool findConnection(KeyFrame* old_kf);
	void computeWindowBRIEFPoint();
	void computeBRIEFPoint();
	//void extractBrief();
	int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
	bool searchInAera(const BRIEF::bitset window_descriptor,
	                  const std::vector<BRIEF::bitset> &descriptors_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old,
	                  const std::vector<cv::KeyPoint> &keypoints_old_norm,
	                  cv::Point2f &best_match,
	                  cv::Point2f &best_match_norm);
	void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
						  std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm);
	void FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                const std::vector<cv::Point2f> &matched_2d_old_norm,
                                vector<uchar> &status);
	void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
	               const std::vector<cv::Point3f> &matched_3d,
	               std::vector<uchar> &status,
	               Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
	void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
	void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
	void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

	Eigen::Vector3d getLoopRelativeT();
	double getLoopRelativeYaw();
	Eigen::Quaterniond getLoopRelativeQ();



    double time_stamp;              // 该帧位姿的时间戳
    int index;                      // 该keyFrame的ID， 只会一直加
	int local_index;
    /// 下面4个变量会根据回环检测进行更新矫正
    Eigen::Vector3d vio_T_w_i;      // 当前estimator节点得到的: 该帧 IMU坐标系 到 世界坐标系的平移
    Eigen::Matrix3d vio_R_w_i;      // 当前estimator节点得到的: 该帧 IMU坐标系 到 世界坐标系的旋转
    Eigen::Vector3d T_w_i;          // pose_graph优化的: 该帧 IMU坐标系 到 世界坐标系的平移
    Eigen::Matrix3d R_w_i;          // pose_graph优化的: 该帧 IMU坐标系 到 世界坐标系的旋转
    /// 下面2个不会变化
    Eigen::Vector3d origin_vio_T;   // 该帧时刻下的: estimator节点得到的: 该帧 IMU坐标系 到 世界坐标系的平移
    Eigen::Matrix3d origin_vio_R;   // 该帧时刻下的: estimator节点得到的: 该帧 IMU坐标系 到 世界坐标系的旋转

    cv::Mat image;                  // 该帧图像的 .clone()拷贝
    cv::Mat thumbnail;              // 该帧图像resize cv::Size(80, 60)
    vector<cv::Point3f> point_3d;   // 最近两帧的特征点 世界坐标
	vector<cv::Point2f> point_2d_uv;
	vector<cv::Point2f> point_2d_norm;
    vector<double> point_id;                // 最近两帧的特征点  id
    vector<cv::KeyPoint> keypoints;         // 新检测的keypoint (fast或者角点)
    vector<cv::KeyPoint> keypoints_norm;    // 新检测的keypoint的归一化平面点
    vector<cv::KeyPoint> window_keypoints;  // 滑动窗口特征点对应的keypoint
    vector<BRIEF::bitset> brief_descriptors;// 新检测的keypoint的描述子
    vector<BRIEF::bitset> window_brief_descriptors; // 滑动窗口特征点对应的描述子
	bool has_fast_point;
	int sequence;

	bool has_loop;
	int loop_index;
    Eigen::Matrix<double, 8, 1 > loop_info;
    /// [0~2] :当前帧 IMU坐标系 到 闭环帧 IMU坐标系 的平移（在闭环帧IMU坐标系的表示）
    /// [3~6] :当前帧 IMU坐标系 到 闭环帧 IMU坐标系 的旋转
    /// [7]   :当前帧 IMU坐标系 到 闭环帧 IMU坐标系 的yaw角旋转
};

