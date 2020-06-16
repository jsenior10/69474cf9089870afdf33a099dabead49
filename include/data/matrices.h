#pragma once
#include "../allIncludes.h"
struct transformStruct
{
	cv::cuda::GpuMat calibration_matrix;
	cv::cuda::GpuMat distance_coefficients;
	cv::Mat transform_matrix;
	cv::Mat inv_transform_matrix;
	cv::Mat cal_out1;
	cv::Mat cal_out2;

	bool transform_calculated;
	bool calibrated;
};