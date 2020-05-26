#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

/**
* @class FeatureTracker
* @Description 视觉前端预处理：对每个相机进行角点LK光流跟踪
*/
class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img,double _cur_time);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    void undistortedPoints();

    cv::Mat mask;//图像掩码
    cv::Mat fisheye_mask;//鱼眼相机mask，用来去除边缘噪点


    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;                          //新一帧提取出来的角点
    /// forw_pts: 光流法跟踪，得到与cur_pts一一对应的点
    /// cur_pts: 与prev_pts一一对应的点
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;    //前前一帧的点，前一帧的点，当前帧的点
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    /// ids[i] 与 cur_pts[i] 是一一对应的
    vector<int> ids;                                    // 当前跟踪的特征点的id，每个特征点的id都是唯一的
                                                        // 在函数FeatureTracker::addPoints()被扩充
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;               // 记录当前帧正在跟踪的特征点 <特征点ID（唯一的），特征点在当前帧的归一化平面坐标>
    map<int, cv::Point2f> prev_un_pts_map;              // 记录上一帧正在跟踪的特征点 <特征点ID（唯一的），特征点在上一帧的归一化平面坐标>
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    static int n_id;//特征点id，每检测到一个新的特征点，就将n_id作为该特征点的id，然后n_id加1
};
