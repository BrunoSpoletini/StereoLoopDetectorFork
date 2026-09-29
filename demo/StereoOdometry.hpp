/**
 * File: StereoOdometry.hpp
 * Description: Minimal stereo visual odometry used to gate loop detection by
 *   travelled distance instead of time (keyframe selection and a distance
 *   based exclusion window). It only estimates the displacement of the
 *   current frame with respect to the last keyframe ("anchor"), so the
 *   odometer does not integrate jitter while the robot is stopped.
 * License: See LICENSE.txt file at the top project folder
**/

#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "StereoParameters.h"

class StereoOdometry
{
public:
  struct Result
  {
    /// true if the current frame became the new keyframe
    bool keyframe = false;
    /// displacement (m) of the current frame w.r.t. the last keyframe
    double displacement = 0;
    /// false if the motion could not be estimated (too few inliers)
    bool tracked = false;
    /// travelled distance (m) accumulated over keyframes, including this one
    double odometer = 0;
  };

  /**
   * @param params rectified stereo parameters (uses the projection matrices)
   * @param step minimum displacement (m) to create a new keyframe
   * @param min_inliers minimum PnP inliers to trust the motion estimate
   */
  StereoOdometry(const StereoParameters &params, double step, int min_inliers = 15)
    : m_step(step), m_min_inliers(min_inliers)
  {
    cv::Mat P1, P2;
    params.left_projection.convertTo(P1, CV_64F);
    params.right_projection.convertTo(P2, CV_64F);
    m_fx = P1.at<double>(0, 0);
    m_fy = P1.at<double>(1, 1);
    m_cx = P1.at<double>(0, 2);
    m_cy = P1.at<double>(1, 2);
    m_bf = -P2.at<double>(0, 3);  // fx * baseline
    m_K = P1.colRange(0, 3).clone();
    m_matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
  }

  /**
   * Processes a stereo frame given its stereo-matched keypoints (keys1[i]
   * matches keys2[i]) and the left descriptors (one row per keypoint).
   */
  Result update(const std::vector<cv::KeyPoint> &keys1,
    const std::vector<cv::KeyPoint> &keys2, const cv::Mat &descs1)
  {
    Result r;
    if (m_anchor_descs.empty())
    {
      setAnchor(keys1, keys2, descs1);
      r.keyframe = true;
      r.odometer = m_odometer;
      return r;
    }

    std::vector<cv::DMatch> matches;
    if (!descs1.empty())
      m_matcher->match(m_anchor_descs, descs1, matches);
    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    for (const auto &m : matches)
    {
      if (m.distance > 50) continue;
      obj.push_back(m_anchor_points[m.queryIdx]);
      img.push_back(keys1[m.trainIdx].pt);
    }

    if ((int)obj.size() >= m_min_inliers)
    {
      cv::Mat rvec, tvec;
      std::vector<int> inliers;
      bool ok = cv::solvePnPRansac(obj, img, m_K, cv::Mat(), rvec, tvec,
        false, 100, 2.0, 0.99, inliers, cv::SOLVEPNP_EPNP);
      if (ok && (int)inliers.size() >= m_min_inliers)
      {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Mat c = -R.t() * tvec;  // current camera center in the anchor frame
        r.displacement = cv::norm(c);
        r.tracked = true;
      }
    }

    // lost tracking: assume we moved enough to start over from this frame
    if (!r.tracked || r.displacement >= m_step)
    {
      m_odometer += r.tracked ? r.displacement : m_step;
      setAnchor(keys1, keys2, descs1);
      r.keyframe = true;
    }
    r.odometer = m_odometer;
    return r;
  }

  double odometer() const { return m_odometer; }

private:
  void setAnchor(const std::vector<cv::KeyPoint> &keys1,
    const std::vector<cv::KeyPoint> &keys2, const cv::Mat &descs1)
  {
    m_anchor_points.clear();
    std::vector<int> rows;
    for (size_t i = 0; i < keys1.size(); i++)
    {
      double d = keys1[i].pt.x - keys2[i].pt.x;
      if (d < 1.0) continue;  // too far or wrong match
      double z = m_bf / d;
      if (z > 40.0) continue;
      m_anchor_points.emplace_back((keys1[i].pt.x - m_cx) * z / m_fx,
        (keys1[i].pt.y - m_cy) * z / m_fy, z);
      rows.push_back(i);
    }
    m_anchor_descs.create(rows.size(), descs1.cols, descs1.type());
    for (size_t j = 0; j < rows.size(); j++)
      descs1.row(rows[j]).copyTo(m_anchor_descs.row(j));
  }

  double m_step;
  int m_min_inliers;
  double m_fx, m_fy, m_cx, m_cy, m_bf;
  cv::Mat m_K;
  cv::Ptr<cv::BFMatcher> m_matcher;
  std::vector<cv::Point3f> m_anchor_points;
  cv::Mat m_anchor_descs;
  double m_odometer = 0;
};
