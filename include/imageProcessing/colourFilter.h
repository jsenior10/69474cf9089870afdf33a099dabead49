#pragma once
#include "../allIncludes.h"

void colour_filter_white(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);

void colour_filter_yellow(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);

void colour_filter_combined(cv::cuda::GpuMat& masked_white, cv::cuda::GpuMat& masked_yellow, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);