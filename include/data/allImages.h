#pragma once
#include "../allIncludes.h"
struct all_images
{
	int erode_noise = 3;
	int dilate = 1;
	cv::Mat frame;
	cv::Mat warp_points;
	cv::Mat line_frame;
	cv::cuda::GpuMat gpu_frame;
	cv::cuda::GpuMat cf_white_output;
	cv::cuda::GpuMat cf_yellow_output;
	cv::cuda::GpuMat cf_combined_output;
	cv::cuda::GpuMat warped;
	cv::cuda::GpuMat hist_out;
	cv::cuda::GpuMat threshold;
	cv::cuda::GpuMat cf_output;
	cv::cuda::GpuMat inv;
	cv::cuda::GpuMat sobel;
	cv::cuda::GpuMat erode_dst;
	cv::cuda::GpuMat resized;
	cv::cuda::GpuMat dilate_dst;
	cv::cuda::GpuMat frame_processed1;
	cv::cuda::GpuMat processed_frame;
	cv::cuda::GpuMat unwarped_frame;
	cv::cuda::GpuMat hsv;
	cv::Mat erosion_element;
	cv::Ptr<cv::cuda::Filter> erode;
};