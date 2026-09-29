/**
 * File: demoDetectorStereo.h
 * Author: Nicolas Soncini (other attributions below)
 * Date: Aug 23, 2024
 * Description: Demo for loop detection on stereo images
 * License: See LICENSE.txt file at the top project folder
 * 
 * This file has been modified from a different version, attribution:
 * Original File: demoDetector.h
 * Original Author: Dorian Galvez-Lopez
 * Original Last Modified Date: -
 *
**/

#ifndef __DEMO_DETECTOR_STEREO__
#define __DEMO_DETECTOR_STEREO__

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/hdf/hdf5.hpp>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h>
#include "DLoopDetector.h"
#include "TemplatedLoopDetector.h"
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

#include "RowMatcher.hpp"
#include "StereoOdometry.hpp"
#include "StereoParameters.h"
#include "yaml-cpp/yaml.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// --------------------------------------------------------------------------

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// @param TVocabulary vocabulary class (e.g: BriefVocabulary)
/// @param TDetector detector class (e.g: BriefLoopDetector)
/// @param TDescriptor descriptor class (e.g: bitset for Brief)
template<class TVocabulary, class TDetector, class TDescriptor>
/// Class to run the demo 
class demoDetectorStereo
{
public:

  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param sparam stereo camera parameters
   */
  demoDetectorStereo(const std::string &vocfile, const std::string &imagedir1,
    const std::string &imagedir2, const std::string &posefile, 
    StereoParameters &stereoparams, bool show, float frequency,
    const YAML::Node &config = YAML::Node(),
    const std::string &output_prefix = "",
    const std::string &cache_path = "");
    
  ~demoDetectorStereo(){}

  /**
   * Runs the demo
   * @param name demo name
   * @param extractor functor to extract features
   */
  void run(const std::string &name, const FeatureExtractor<TDescriptor> &extractor);

protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
  void readPoseFile(const char *filename, std::vector<double> &xs, 
    std::vector<double> &ys) const;

  vector<string> read_file_list(std::string path) const;

  void readPoseFileCSV(
    const char *filename, std::vector<double> &xs, std::vector<double> &ys) const;

protected:

  std::string m_vocfile;
  std::string m_imagedir1;
  std::string m_imagedir2;
  std::string m_posefile;
  StereoParameters m_stereoparams;
  bool m_show;
  float m_frequency;
  YAML::Node m_config;
  std::string m_output_prefix;
  std::string m_cache_path;
};

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
demoDetectorStereo<TVocabulary, TDetector, TDescriptor>::demoDetectorStereo
  (const std::string &vocfile,
   const std::string &imagedir1, const std::string &imagedir2,
   const std::string &posefile, StereoParameters &stereoparams, 
   bool show, float frequency, const YAML::Node &config,
   const std::string &output_prefix, const std::string &cache_path)
  : m_vocfile(vocfile), m_imagedir1(imagedir1), m_imagedir2(imagedir2),
    m_posefile(posefile), m_stereoparams(stereoparams), m_show(show),
    m_frequency(frequency), m_config(config), m_output_prefix(output_prefix),
    m_cache_path(cache_path)
{
}

// ---------------------------------------------------------------------------

/// Reads `key` from the config node, or returns `def` if it is not there
template<typename T>
T cfg(const YAML::Node &node, const std::string &key, const T &def)
{
  if (node && node[key]) return node[key].as<T>();
  return def;
}

// ---------------------------------------------------------------------------

// Binary feature cache: for every image, the raw keypoints and descriptors
// of both stereo images (before stereo matching), so that runs that only
// change the detection stage do not recompute ORB.
namespace FeatureCache
{
  inline void writeKeys(std::ostream &o, const std::vector<cv::KeyPoint> &keys,
    const cv::Mat &descs)
  {
    int n = keys.size(), cols = descs.cols;
    o.write((char*)&n, sizeof(int));
    o.write((char*)&cols, sizeof(int));
    for (const auto &k : keys) {
      float f[5] = {k.pt.x, k.pt.y, k.size, k.angle, k.response};
      o.write((char*)f, sizeof(f));
      o.write((char*)&k.octave, sizeof(int));
    }
    for (int i = 0; i < n; i++)
      o.write((char*)descs.ptr(i), cols);
  }

  inline bool readKeys(std::istream &in, std::vector<cv::KeyPoint> &keys,
    cv::Mat &descs)
  {
    int n, cols;
    if (!in.read((char*)&n, sizeof(int))) return false;
    in.read((char*)&cols, sizeof(int));
    keys.resize(n);
    for (auto &k : keys) {
      float f[5];
      in.read((char*)f, sizeof(f));
      in.read((char*)&k.octave, sizeof(int));
      k.pt = cv::Point2f(f[0], f[1]); k.size = f[2]; k.angle = f[3];
      k.response = f[4];
    }
    descs.create(n, cols, CV_8U);
    for (int i = 0; i < n; i++)
      in.read((char*)descs.ptr(i), cols);
    return (bool)in;
  }
}

// ---------------------------------------------------------------------------

// Function to convert std::bitset to cv::Mat
cv::Mat toMat(const std::vector<std::bitset<256>>& descriptors) {
    // Assuming each bitset<256> corresponds to one descriptor
    int descriptorCount = descriptors.size();
    int bitsPerDescriptor = 256;

    // Create an empty Mat to store the descriptors
    cv::Mat descriptorsMat(descriptorCount, bitsPerDescriptor, CV_8UC1);

    // Convert each bitset to Mat
    for (int i = 0; i < descriptorCount; ++i) {
        for (int j = 0; j < bitsPerDescriptor; ++j) {
            descriptorsMat.at<uchar>(i, j) = descriptors[i][j] ? 255 : 0;
        }
    }

    return descriptorsMat;
}

cv::Mat toMat(const std::vector<cv::Mat>& descriptors){
  int descriptorCount = descriptors.size();
  cv::Mat descriptorsMat;

  for (int i = 0; i < descriptorCount; ++i) {
    descriptorsMat.push_back(descriptors.at(i));
  }

  return descriptorsMat;
}

// ---------------------------------------------------------------------------

// Function to extract matching elements from two arrays
template<typename T>
void extractMatchingElements(const std::vector<cv::DMatch>& matches,
                             const std::vector<T>& array1,
                             const std::vector<T>& array2,
                             std::vector<T>& matchingElements1,
                             std::vector<T>& matchingElements2) {
    matchingElements1.clear();
    matchingElements2.clear();

    // Iterate through the matches
    for (const auto& match : matches) {
        // Get indices from the match
        int index1 = match.queryIdx;
        int index2 = match.trainIdx;

        // Add matching elements from array1 and array2 to respective vectors
        matchingElements1.push_back(array1[index1]);
        matchingElements2.push_back(array2[index2]);
    }

}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetectorStereo<TVocabulary, TDetector, TDescriptor>::run
  (const std::string &name, const FeatureExtractor<TDescriptor> &extractor)
{ 
  // Set loop detector parameters
  // Some references for frequency:
  //   EuRoC 20fps, FieldSAFE stereo 10fps / webcam 30fps
  int m_height = m_stereoparams.size.height;
  int m_width = m_stereoparams.size.width;
  typename TDetector::Parameters params(m_height, m_width, m_frequency);
  
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0
  
  // We are going to change these values individually. Every value can be
  // overridden from the YAML config (--config); the defaults reproduce the
  // original configuration of the paper.
  const YAML::Node &c = m_config;
  params.use_nss = cfg(c, "use_nss", true); // use normalized similarity instead of raw score
  params.alpha = cfg(c, "alpha", 0.3f); // nss threshold
  params.k = cfg(c, "k", 5); // a loop must be consistent with k previous matches
  params.geom_check = GEOM_EXHAUSTIVE_STEREO;
  params.stereo_params = m_stereoparams;
  params.di_levels = 2; // number of direct index levels
  // NOTE: these used to be ints, so the original 0.4 was truncated to 0
  params.near_distance = cfg(c, "near_distance", 0.0); // min meters for triangulated points
  params.far_distance = cfg(c, "far_distance", 50.0); // max meters for triangulated points 
  // NO tocar params.dislocal: Parameters(h, w, frequency) ya lo deja en
  // 20*frequency, o sea los 20 segundos que usa el paper, y ademas deriva de
  // la frecuencia otros 5 parametros que deben quedar coherentes entre si.
  params.min_Fpoints = cfg(c, "min_Fpoints", 20); // min points to compute fundamental matrix 
  params.max_neighbor_ratio = cfg(c, "max_neighbor_ratio", 0.6);
  params.max_db_results = cfg(c, "max_db_results", params.max_db_results);
  params.pnp_rectified_intrinsics = cfg(c, "pnp_rectified_intrinsics", false);
  params.pnp_ransac = cfg(c, "pnp_ransac", false);
  params.pnp_reprojection_error = cfg(c, "pnp_reprojection_error", 3.0);
  params.min_pnp_inliers = cfg(c, "min_pnp_inliers", params.min_Fpoints);
  params.stereo_cross_check = cfg(c, "stereo_cross_check", true);
  params.require_positive_depth = cfg(c, "require_positive_depth", false);
  params.exclusion_distance = cfg(c, "exclusion_distance", 0.0);
  params.geom_candidates = cfg(c, "geom_candidates", 1);
  params.strong_inliers = cfg(c, "strong_inliers", 0);
  // motion gating with stereo visual odometry (0 disables it)
  const double motion_step = cfg(c, "motion_step", 0.0);
  const bool keyframes_only = cfg(c, "keyframes_only", false);
  StereoOdometry odometry(m_stereoparams, motion_step > 0 ? motion_step : 1.0);
  // image index of every entry added to the detector database
  std::vector<int> entry_image;
  
  // To verify loops you can select one of the next geometrical checkings:
  // GEOM_EXHAUSTIVE_STEREO: correspondence points are computed by comparing 
  //    all the features between the two stereo pairs, keeping the ones that
  //    match between them and performing PnP to return a transformation
  //    between the triangulated 3D points of the first pair.
  // GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
  //    which makes them faster. However, creating the flann structure may
  //    be slow.
  // GEOM_DI: the direct index is used to select correspondence points between
  //    those features whose vocabulary node at a certain level is the same.
  //    The level at which the comparison is done is set by the parameter
  //    di_levels:
  //      di_levels = 0 -> features must belong to the same leaf (word).
  //         This is the fastest configuration and the most restrictive one.
  //      di_levels = l (l < L) -> node at level l starting from the leaves.
  //         The higher l, the slower the geometrical checking, but higher
  //         recall as well.
  //         Here, L stands for the depth levels of the vocabulary tree.
  //      di_levels = L -> the same as the exhaustive technique.
  // GEOM_NONE: no geometrical checking is done.
  //
  // In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4 
  // yields the best results in recall/time.
  // Check the T-RO paper for more information.
  //
  
  // Load the vocabulary to use
  cout << "Loading " << name << " vocabulary..." << endl;
  TVocabulary voc(m_vocfile);
  cout << "... done" << endl;
  
  // Initiate loop detector with the vocabulary 
  cout << "Initializing detector..." << endl;
  TDetector detector(voc, params);
  cout << "... done" << endl;

  // Initialize row keypoint matcher
  // TODO: if we decide to make SURF descriptors available this would need to
  // change to NORM_L2 or similar, maybe passing the stereo matcher as a 
  // parameter. Maybe passing the stereo matcher already initialized in the 
  // parameters?
  cv::Ptr<cv::BFMatcher> descriptor_matcher = 
    cv::BFMatcher::create(cv::NORM_HAMMING);
  double max_distance = cfg(c, "stereo_max_descriptor_distance", 50.0);
  double row_range = cfg(c, "stereo_row_range", 1.0);
  // disparity limits (px) for stereo matches; the defaults disable the check
  double min_disparity = cfg(c, "stereo_min_disparity", -1e9);
  double max_disparity = cfg(c, "stereo_max_disparity", 1e9);
  RowMatcher stereo_matcher(max_distance, descriptor_matcher, row_range);
  
  // Process images
  std::vector<cv::KeyPoint> keys1, keys2;
  std::vector<TDescriptor> descriptors1, descriptors2;

  // load image filenames
  cout << "Collecting image paths..." << endl;
  std::vector<string> filenames1 =
    read_file_list(m_imagedir1);
    // DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".png", true);
  std::vector<string> filenames2 = 
    read_file_list(m_imagedir2);
    // DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".png", true);
  assert(filenames1.size() == filenames2.size());
  cout << "... done" << endl;
  
  // load robot poses
  cout << "Loading poses..." << endl;
  std::vector<double> xs, ys;
  // readPoseFile(m_posefile.c_str(), xs, ys);
  readPoseFileCSV(m_posefile.c_str(), xs, ys);
  cout << "done..." << endl;
  
  // we can allocate memory for the expected number of images
  detector.allocate(filenames1.size());
  
  // prepare visualization windows
  DUtilsCV::GUI::tWinHandler win = "Current image";
  DUtilsCV::GUI::tWinHandler winplot = "Trajectory";
  
  DUtilsCV::Drawing::Plot::Style normal_style(2); // thickness
  DUtilsCV::Drawing::Plot::Style loop_style('r', 2); // color, thickness
  
  DUtilsCV::Drawing::Plot implot(240, 320,
    - *std::max_element(xs.begin(), xs.end()),
    - *std::min_element(xs.begin(), xs.end()),
    *std::min_element(ys.begin(), ys.end()),
    *std::max_element(ys.begin(), ys.end()), 20);
  
  // prepare profiler to measure times
  DUtils::Profiler profiler;

  // Detected loops: query image id, matched image id, and the estimated
  // translation in the query camera frame (x right, y down, z forward)
  std::vector<int> loop_query_ids;
  std::vector<int> loop_match_ids;
  std::vector<float> loop_translation_x;
  std::vector<float> loop_translation_y;
  std::vector<float> loop_translation_z;

  // Create a file to save detection info
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream dtime;
  dtime << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S");
  std::string prefix = m_output_prefix.empty() ? dtime.str() : m_output_prefix;
  std::string fstorepath = prefix + "_results.yml";
  cv::FileStorage fstore(fstorepath, cv::FileStorage::WRITE);
  // one row per query with the diagnostics of the detection
  std::ofstream qlog(prefix + "_queries.csv");
  qlog << "query,status,candidate,ns_factor,best_score,consistent,"
          "geom_matches,geom_inliers,n_stereo,tx,ty,tz,rx,ry,rz,odometer\n";
  std::vector<float> loop_rotation_x, loop_rotation_y, loop_rotation_z;

  // feature cache (ORB only): read it if it exists, write it otherwise
  std::ifstream cache_in;
  std::ofstream cache_out;
  constexpr bool cacheable = std::is_same<TDescriptor, cv::Mat>::value;
  if (cacheable && !m_cache_path.empty()) {
    cache_in.open(m_cache_path, std::ios::binary);
    if (cache_in.is_open())
      cout << "Reading features from cache " << m_cache_path << endl;
    else {
      cache_out.open(m_cache_path + ".tmp", std::ios::binary);
      cout << "Writing features to cache " << m_cache_path << endl;
    }
  }

  int count = 0;
  
  // go
  for(unsigned int i = 0; i < filenames1.size(); ++i)
  {
    std::cout << "Adding image " << i << std::endl;
    
    // get features
    profiler.profile("features");
    bool from_cache = false;
    if constexpr (cacheable) {
      if (cache_in.is_open()) {
        cv::Mat d1, d2;
        if (!FeatureCache::readKeys(cache_in, keys1, d1) ||
            !FeatureCache::readKeys(cache_in, keys2, d2))
          throw std::string("feature cache ended before the image list");
        descriptors1.resize(d1.rows);
        for (int r = 0; r < d1.rows; r++) descriptors1[r] = d1.row(r);
        descriptors2.resize(d2.rows);
        for (int r = 0; r < d2.rows; r++) descriptors2[r] = d2.row(r);
        from_cache = true;
      }
    }
    if (!from_cache) {
      cv::Mat im1 = cv::imread(filenames1[i].c_str(), 0);
      cv::Mat im2 = cv::imread(filenames2[i].c_str(), 0);
      if(m_show)
        DUtilsCV::GUI::showImage(im1, true, &win, 10);
      extractor(im1, keys1, descriptors1);
      extractor(im2, keys2, descriptors2);
      if constexpr (cacheable) {
        if (cache_out.is_open()) {
          FeatureCache::writeKeys(cache_out, keys1, toMat(descriptors1));
          FeatureCache::writeKeys(cache_out, keys2, toMat(descriptors2));
        }
      }
    }
    profiler.stop();
    std::cout << "[StereoMatcher] found " << keys1.size() << " and " <<
      keys2.size() << " keypoints" << std::endl;

    // We should only pass matching key/descriptors to detectLoop stereo
    std::vector<cv::DMatch> stereo_matches;
    // transforms descriptors to cv::Mat for compatibility
    cv::Mat mat_descriptors1 = toMat(descriptors1);
    cv::Mat mat_descriptors2 = toMat(descriptors2);
    stereo_matcher.match(
      keys1, mat_descriptors1, keys2, mat_descriptors2, stereo_matches
    );
    // keep only matches with a plausible disparity (x_left - x_right)
    stereo_matches.erase(std::remove_if(stereo_matches.begin(),
      stereo_matches.end(), [&](const cv::DMatch &m) {
        double d = keys1[m.queryIdx].pt.x - keys2[m.trainIdx].pt.x;
        return d < min_disparity || d > max_disparity;
      }), stereo_matches.end());
    std::cout << "[StereoMatcher] found " << stereo_matches.size() 
      << " stereo matches" << std::endl; 
    // reorder keypoints and descriptors to match in order
    std::vector<cv::KeyPoint> s_keys1, s_keys2;
    extractMatchingElements<cv::KeyPoint>(
      stereo_matches, keys1, keys2, s_keys1, s_keys2
    );
    std::vector<TDescriptor> s_descriptors1, s_descriptors2;
    extractMatchingElements<TDescriptor>(
      stereo_matches, descriptors1, descriptors2, s_descriptors1, s_descriptors2
    );
    
    // motion gating: travelled distance and keyframe selection
    double odometer = -1;
    bool process = true;
    if (motion_step > 0)
    {
      StereoOdometry::Result mo =
        odometry.update(s_keys1, s_keys2, toMat(s_descriptors1));
      odometer = mo.odometer;
      process = mo.keyframe || !keyframes_only;
    }

    // add image to the collection and check if there is some loop
    DetectionResult result;
    if (!process)
    {
      qlog << i << ",-1,-1,-1,-1,0,-1,-1," << s_keys1.size()
        << ",0,0,0,0,0,0," << odometer << "\n";
      continue;
    }
    
    profiler.profile("detection");
    detector.detectLoop(
        s_keys1, s_descriptors1, result, s_keys2, s_descriptors2, odometer);
    profiler.stop();
    entry_image.push_back((int)i);
    if (result.status >= NO_GROUPS || result.status == LOOP_DETECTED)
      result.match = entry_image[result.match];
    
    if(result.detection())
    {
      cout << "- Loop found with image " << result.match << "!" << endl;
      cout << "- \t with translation:  " << result.transform << endl;
      ++count;

      loop_query_ids.push_back((int) i);
      loop_match_ids.push_back((int) result.match);
      loop_translation_x.push_back(static_cast<float>(result.transform[0]));
      loop_translation_y.push_back(static_cast<float>(result.transform[1]));
      loop_translation_z.push_back(static_cast<float>(result.transform[2]));
      loop_rotation_x.push_back(static_cast<float>(result.rotation[0]));
      loop_rotation_y.push_back(static_cast<float>(result.rotation[1]));
      loop_rotation_z.push_back(static_cast<float>(result.rotation[2]));
    }
    else
    {
      cout << "- No loop: ";
      switch(result.status)
      {
        case CLOSE_MATCHES_ONLY:
          cout << "All the images in the database are very recent" << endl;
          break;
          
        case NO_DB_RESULTS:
          cout << "There are no matches against the database (few features in"
            " the image?)" << endl;
          break;
          
        case LOW_NSS_FACTOR:
          cout << "Little overlap between this image and the previous one"
            << endl;
          break;
            
        case LOW_SCORES:
          cout << "No match reaches the score threshold (alpha: " <<
            params.alpha << ")" << endl;
          break;
          
        case NO_GROUPS:
          cout << "Not enough close matches to create groups. "
            << "Best candidate: " << result.match << endl;
          break;
          
        case NO_TEMPORAL_CONSISTENCY:
          cout << "No temporal consistency (k: " << params.k << "). "
            << "Best candidate: " << result.match << endl;
          break;
          
        case NO_GEOMETRICAL_CONSISTENCY:
          cout << "No geometrical consistency. Best candidate: " 
            << result.match << endl;
          break;
          
        default:
          break;
      }
    }
    
    cout << endl;

    qlog << i << "," << (int)result.status << "," 
      << (result.status >= NO_GROUPS || result.status == LOOP_DETECTED ?
          (int)result.match : -1) << ","
      << result.ns_factor << "," << result.best_score << "," 
      << result.consistent << "," << result.geom_matches << ","
      << result.geom_inliers << "," << s_keys1.size() << ","
      << result.transform[0] << "," << result.transform[1] << ","
      << result.transform[2] << "," << result.rotation[0] << ","
      << result.rotation[1] << "," << result.rotation[2] << ","
      << odometer << "\n";
    
    // show trajectory
    if(m_show && i > 0)
    {
      if(result.detection())
        implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], loop_style);
      else
        implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], normal_style);
      
      DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10); 
    }
  }

  fstore << "num_images" << (int) filenames1.size();
  fstore << "num_loops" << count;
  fstore << "loop_query_ids" << loop_query_ids;
  fstore << "loop_match_ids" << loop_match_ids;
  fstore << "loop_translation_x" << loop_translation_x;
  fstore << "loop_translation_y" << loop_translation_y;
  fstore << "loop_translation_z" << loop_translation_z;
  fstore << "loop_rotation_x" << loop_rotation_x;
  fstore << "loop_rotation_y" << loop_rotation_y;
  fstore << "loop_rotation_z" << loop_rotation_z;
  fstore << "feature_time_ms" << profiler.getMeanTime("features") * 1e3;
  fstore << "loop_detection_time_ms" << profiler.getMeanTime("detection") * 1e3;
  fstore.release();
  qlog.close();
  if (cache_out.is_open()) {
    cache_out.close();
    std::rename((m_cache_path + ".tmp").c_str(), m_cache_path.c_str());
  }

  if(count == 0)
  {
    cout << "No loops found in this image sequence" << endl;
  }
  else
  {
    cout << count << " loops found in this image sequence!" << endl;
  }

  cout << endl << "Execution time:" << endl
    << " - Feature computation: " << profiler.getMeanTime("features") * 1e3
    << " ms/image" << endl
    << " - Loop detection: " << profiler.getMeanTime("detection") * 1e3
    << " ms/image" << endl;

  if(m_show) {
    cout << endl << "Press a key to finish..." << endl;
    DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 0);
  }
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetectorStereo<TVocabulary, TDetector, TDescriptor>::readPoseFile
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys)
  const
{
  xs.clear();
  ys.clear();
  
  fstream f(filename, ios::in);
  
  string s;
  double ts, x, y, t;
  while(!f.eof())
  {
    getline(f, s);
    if(!f.eof() && !s.empty())
    {
      sscanf(s.c_str(), "%lf, %lf, %lf, %lf", &ts, &x, &y, &t);
      xs.push_back(x);
      ys.push_back(y);
    }
  }
  
  f.close();
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetectorStereo<TVocabulary, TDetector, TDescriptor>::readPoseFileCSV
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys)
  const
{
  // ASSUMES CSV HEADERS: 'nanoseconds,x,y,ow,ox,oy,oz'
  xs.clear();
  ys.clear();
  
  fstream f(filename, ios::in);
  
  string s;
  // double ts, x, y, t;
  double x, y;
  while(!f.eof())
  {
    getline(f, s);
    if(!f.eof() && !s.empty())
    {
      sscanf(s.c_str(), "%*d,%lf,%lf,%*s", &x, &y);
      xs.push_back(x);
      ys.push_back(y);
    }
  }
  
  f.close();
}

// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor> 
std::vector<std::string> 
demoDetectorStereo<TVocabulary, TDetector, TDescriptor>::read_file_list
  (std::string path)
  const
{
    std::ifstream file;
    std::string line;
    std::vector<std::string> list;

    file.open(path);
    assert(file.is_open());
    while (getline(file, line)) {
        list.push_back(line);
    }
    return list;
}

// ---------------------------------------------------------------------------

#endif
