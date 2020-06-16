#pragma once
#include "../allIncludes.h"
struct colour_boundaries
{
	cv::Scalar lower_yellow;
	cv::Scalar upper_yellow;

	cv::Scalar lower_white;
	cv::Scalar upper_white;

	const int GRADIENT_THRES_MIN = 30;
	const int GRADIENT_THRES_MAX = 100;
	const int COLOUR_THRES_MIN = 180;
	const int COLOUR_THRES_MAX = 255;
	const int OFFSET = 100;
};