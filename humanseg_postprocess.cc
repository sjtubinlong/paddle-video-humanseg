#include "humanseg_postprocess.h"
#include <iostream>
#include <string.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

typedef unsigned char uch;

namespace human_seg {

cv::Mat prev_fg_cfd;
cv::Mat cur_fg_cfd;
cv::Mat cur_fg_mask;
cv::Mat track_fg_cfd;
cv::Mat prev_gray;
cv::Mat cur_gray;
cv::Mat bgr_temp;
cv::Mat is_track;
cv::Mat static_roi;
cv::Mat weights;
cv::Ptr<cv::optflow::DISOpticalFlow> disflow = \
    cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);

bool is_init = false;

int human_seg_track_fuse(const cv::Mat &track_fg_cfd, const cv::Mat &dl_fg_cfd,
	cv::Mat &cur_fg_cfd, const cv::Mat &dl_weights, const cv::Mat &is_track,
	const float cfd_diff_thres, const int patch_size) {

	float *cur_fg_cfd_ptr = (float *)cur_fg_cfd.data;
	float *dl_fg_cfd_ptr = (float *)dl_fg_cfd.data;
	float *track_fg_cfd_ptr = (float *)track_fg_cfd.data;
	float *dl_weights_ptr = (float *)dl_weights.data;
	uchar *is_track_ptr = (uchar *)is_track.data;
	int y_offset = 0;
	int ptr_offset = 0;
	int h = track_fg_cfd.rows;
	int w = track_fg_cfd.cols;
	float dl_fg_score = 0.0;
	float track_fg_score = 0.0;
	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			dl_fg_score = dl_fg_cfd_ptr[ptr_offset];
			if (is_track_ptr[ptr_offset] > 0) {
				track_fg_score = track_fg_cfd_ptr[ptr_offset];
				if (dl_fg_score > 0.9 || dl_fg_score < 0.1) {
					if (dl_weights_ptr[ptr_offset] <= 0.10) {
						cur_fg_cfd_ptr[ptr_offset] = dl_fg_score * 0.3 + track_fg_score * 0.7;
					}
					else {
						cur_fg_cfd_ptr[ptr_offset] = dl_fg_score * 0.4 + track_fg_score * 0.6;
					}
				}
				else {
					cur_fg_cfd_ptr[ptr_offset] = dl_fg_score * dl_weights_ptr[ptr_offset]
						+ track_fg_score * (1 - dl_weights_ptr[ptr_offset]);
				}
			}
			else {
				cur_fg_cfd_ptr[ptr_offset] = dl_fg_score;
			}
			++ptr_offset;
		}
		y_offset += w;
		ptr_offset = y_offset;
	}
return 0;
}


int human_seg_tracking(const cv::Mat &prev_gray, const cv::Mat &cur_gray,
       const cv::Mat &prev_fg_cfd, cv::Mat &track_fg_cfd,
       cv::Mat &is_track, cv::Mat &dl_weights,
       cv::Ptr<cv::optflow::DISOpticalFlow> disflow,
       const int patch_size) {

	cv::Mat flow_fw;
	disflow->calc(prev_gray, cur_gray, flow_fw);

	cv::Mat flow_bw;
	disflow->calc(cur_gray, prev_gray, flow_bw);

	float double_check_thres = 8;

	cv::Point2f fxy_fw;
	int dy_fw = 0;
	int dx_fw = 0;
	cv::Point2f fxy_bw;
	int dy_bw = 0;
	int dx_bw = 0;

	float *prev_fg_cfd_ptr = (float *)prev_fg_cfd.data;
	float *track_fg_cfd_ptr = (float *)track_fg_cfd.data;
	float *dl_weights_ptr = (float *)dl_weights.data;
	uchar *is_track_ptr = (uchar *)is_track.data;

	int prev_y_offset = 0;
	int prev_ptr_offset = 0;
	int cur_ptr_offset = 0;
	float *flow_fw_ptr = (float *)flow_fw.data;

	float roundy_fw = 0.0;
	float roundx_fw = 0.0;
	float roundy_bw = 0.0;
	float roundx_bw = 0.0;

	int h = prev_fg_cfd.rows;
	int w = prev_fg_cfd.cols;
	for (int r = 0; r < h; ++r) {
		for (int c = 0; c < w; ++c) {
			++prev_ptr_offset;

			fxy_fw = flow_fw.ptr<cv::Point2f>(r)[c];
			//            dy_fw = roundf(fxy_fw.y);
			//            dx_fw = roundf(fxy_fw.x);
			roundy_fw = fxy_fw.y >= 0 ? 0.5 : -0.5;
			roundx_fw = fxy_fw.x >= 0 ? 0.5 : -0.5;
			dy_fw = (int)(fxy_fw.y + roundy_fw);
			dx_fw = (int)(fxy_fw.x + roundx_fw);

			int cur_x = c + dx_fw;
			int cur_y = r + dy_fw;

			if (cur_x < 0 || cur_x >= h || cur_y < 0
				|| cur_y >= w) {
				continue;
			}

			fxy_bw = flow_bw.ptr<cv::Point2f>(cur_y)[cur_x];
			//            dy_bw = roundf(fxy_bw.y);
			//            dx_bw = roundf(fxy_bw.x);
			roundy_bw = fxy_bw.y >= 0 ? 0.5 : -0.5;
			roundx_bw = fxy_bw.x >= 0 ? 0.5 : -0.5;
			dy_bw = (int)(fxy_bw.y + roundy_bw);
			dx_bw = (int)(fxy_bw.x + roundx_bw);

			if (((dy_fw + dy_bw) * (dy_fw + dy_bw) + (dx_fw + dx_bw) * (dx_fw + dx_bw))
				>= double_check_thres) {
				continue;
			}

			cur_ptr_offset = cur_y * w + cur_x;


			if (abs(dy_fw) <= 0 && abs(dx_fw) <= 0 && abs(dy_bw) <= 0 && abs(dx_bw) <= 0) {
				dl_weights_ptr[cur_ptr_offset] = 0.05;
			}

			is_track_ptr[cur_ptr_offset] = 1; 
			track_fg_cfd_ptr[cur_ptr_offset] = prev_fg_cfd_ptr[prev_ptr_offset];

		}
		prev_y_offset += w;
		prev_ptr_offset = prev_y_offset - 1;
	}

	return 0;
}
 


int humanseg_postprocess(const unsigned char *bgr_img, const float *dl_output,
                         const int height, const int width, unsigned char *seg_result) {

    const float *cfd_ptr = dl_output;
    
    if (!is_init) {
        is_init = true;
        cur_fg_cfd = cv::Mat(height, width, CV_32FC1, cv::Scalar::all(0));
        memcpy(cur_fg_cfd.data, cfd_ptr, height * width * sizeof(float));
        cur_fg_mask = cv::Mat(height, width, CV_8UC1, cv::Scalar::all(0));

        if (height <= 64 || width <= 64) {
            disflow->setFinestScale(1);
        } else if (height <= 160 || width <= 160) {
            disflow->setFinestScale(2);
        } else {
            disflow->setFinestScale(3);
        }
        is_track = cv::Mat(height, width, CV_8UC1, cv::Scalar::all(0));
        static_roi = cv::Mat(height, width, CV_8UC1, cv::Scalar::all(0));
        track_fg_cfd = cv::Mat(height, width, CV_32FC1, cv::Scalar::all(0));

        bgr_temp = cv::Mat(height, width, CV_8UC3);
        memcpy(bgr_temp.data, bgr_img, height * width * 3 * sizeof(uch));
        cv::cvtColor(bgr_temp, cur_gray, cv::COLOR_BGR2GRAY);
        weights = cv::Mat(height, width, CV_32FC1, cv::Scalar::all(0.30));

    } else {
        memcpy(cur_fg_cfd.data, cfd_ptr, height * width * sizeof(float));
        memcpy(bgr_temp.data, bgr_img, height * width * 3 * sizeof(uch));
        cv::cvtColor(bgr_temp, cur_gray, cv::COLOR_BGR2GRAY);
        memset(is_track.data, 0, height * width * sizeof(uch));
        memset(static_roi.data, 0, height * width * sizeof(uch));
        weights = cv::Mat(height, width, CV_32FC1, cv::Scalar::all(0.30));
        int is_static = human_seg_tracking(prev_gray, cur_gray,
                                                 prev_fg_cfd, track_fg_cfd,
                                                 is_track, weights, disflow, 0);

        human_seg_track_fuse(track_fg_cfd, cur_fg_cfd, cur_fg_cfd, weights,
                                   is_track, 1.1, 0);

    }
    
    int ksize = 3;
    cv::GaussianBlur(cur_fg_cfd, cur_fg_cfd, cv::Size(ksize, ksize), 0, 0);
    prev_fg_cfd = cur_fg_cfd.clone();
    prev_gray = cur_gray.clone();
    
    cur_fg_cfd.convertTo(cur_fg_mask, CV_8UC1, 255);
    memcpy(seg_result, cur_fg_mask.data, height * width);

    return 0;
}

cv::Mat save_seg_res(cv::Mat seg_mat, cv::Mat ori_frame){
    cv::Mat return_frame;
    cv::resize(ori_frame, return_frame, cv::Size(ori_frame.cols, ori_frame.rows));
	for (int i = 0; i < ori_frame.rows; i++) {
        for (int j = 0; j < ori_frame.cols; j++) {
            float score = seg_mat.at<uchar>(i,j)/255.0;
            if (score > 0.1 ) {
                 return_frame.at<cv::Vec3b>(i, j)[2] = (int)((1-score)*255 + score*return_frame.at<cv::Vec3b>(i, j)[2]);
                 return_frame.at<cv::Vec3b>(i, j)[1] = (int)((1-score)*255 + score*return_frame.at<cv::Vec3b>(i, j)[1]);
                 return_frame.at<cv::Vec3b>(i, j)[0] = (int)((1-score)*255 + score*return_frame.at<cv::Vec3b>(i, j)[0]);
            }
            else{
                return_frame.at<cv::Vec3b>(i, j) = {255,255,255};
            }
        }
     }
     return return_frame;
}


int threshold_mask(const cv::Mat &fg_cfd, float fg_thres, float bg_thres,
                   cv::Mat &fg_mask) {
    if (fg_cfd.type() != CV_32FC1) {
        std::cout << "threshold_mask: type is not CV_32FC1"<<std::endl;
        return -1;
    }
    if (!(fg_mask.type() == CV_8UC1 && fg_mask.rows == fg_cfd.rows
                  && fg_mask.cols == fg_cfd.cols)) {
        fg_mask = cv::Mat(fg_cfd.rows, fg_cfd.cols, CV_8UC1, cv::Scalar::all(0));
    }

    for (int r = 0; r < fg_cfd.rows; ++r) {
        for (int c = 0; c < fg_cfd.cols; ++c) {
            float score = fg_cfd.at<float>(r, c);
            if (score < bg_thres) {
                fg_mask.at<uch>(r, c) = 0;
            } else if (score > fg_thres) {
                fg_mask.at<uch>(r, c) = 255;
            } else {
                fg_mask.at<uch>(r, c) = static_cast<int>((score-bg_thres) /
                                                          (fg_thres - bg_thres) * 255);
            }
        }
    }
    return 0;
}


}

