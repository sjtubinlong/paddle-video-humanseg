
#ifndef HUMANSEG_POSTPROCESS_H
#define HUMANSEG_POSTPROCESS_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>


namespace human_seg {

int humanseg_postprocess(const unsigned char *bgr_img,
                         const float *dl_output,
                         const int height,
                         const int width,
                         unsigned char *seg_result);


int threshold_mask(const cv::Mat &fg_cfd, float fg_thres, float bg_thres,
                   cv::Mat &fg_mask);

cv::Mat save_seg_res(cv::Mat seg_mat, cv::Mat ori_frame);
}

#endif // HUMANSEG_POSTPROCESS_H
