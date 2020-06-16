#include "../../include/allIncludes.h"

void LaneDetector::colour_filter_white(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream)
{
	cv::cuda::GpuMat hsv_image(720, 1280, CV_8UC1);
	cv::cuda::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV, 0, stream); //async yeeeeee boy
	//stream.waitForCompletion();

	custom_cuda::in_range_gpu(hsv_image, cb.lower_white, cb.upper_white, output_image);
}

void LaneDetector::colour_filter_yellow(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream)
{
	cv::cuda::GpuMat hsv_image;
	cv::cuda::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV, 0, stream);
	//stream.waitForCompletion();

	custom_cuda::in_range_gpu(hsv_image, cb.lower_yellow, cb.upper_yellow, output_image);
}

void LaneDetector::colour_filter_combined(cv::cuda::GpuMat& masked_white, cv::cuda::GpuMat& masked_yellow, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream)
{
	cv::cuda::bitwise_or(masked_white, masked_yellow, output_image, cv::noArray(), stream);
}

void LaneDetector::initialise_colour_boundaries()
{
	cb.lower_white = cv::Scalar(20, 0, 200);
	cb.upper_white = cv::Scalar(255, 80, 255);

	cb.lower_yellow = cv::Scalar(0, 80, 200);
	cb.upper_white = cv::Scalar(40, 255, 255);
}