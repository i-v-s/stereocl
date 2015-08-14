#ifndef STEREO_MATCH_H
#define STEREO_MATCH_H
#include <opencv2/core.hpp>

class OCVStereo
{
private:
    cv::Mat M1, D1, M2, D2;
    cv::Mat R, T, R1, P1, R2, P2, Q;
    cv::Mat map11, map12, map21, map22;
    float mScale;
public:
    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2, STEREO_VAR=3 };
    OCVStereo(float scale = 1.f, const char * intrinsic_filename = "ocvCalib/intrinsics.yml", const char * extrinsic_filename = "ocvCalib/extrinsics.yml");
    void rectify(cv::Mat &img1, cv::Mat &img2);
};

#endif // STEREO_MATCH_H
