#pragma once
#include "../allIncludes.h"

void LaneDetector::sobel_filter(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image);