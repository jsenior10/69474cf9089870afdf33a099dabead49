#include "../../include/allIncludes.h"

void LaneDetector::sobel_filter(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image)
{
	cv::cuda::GpuMat sobelx, sobely, adwsobelx, adwsobely, gray_framea;
	cv::cuda::cvtColor(input_image, gray_framea, cv::COLOR_BGR2HSV);
	cv::Ptr<cv::cuda::Filter> filter;
	filter = cv::cuda::createSobelFilter(gray_framea.type(), CV_16S, 1, 0, 3, 1,
		cv::BORDER_DEFAULT);
	filter->apply(gray_framea, sobelx);
	cv::cuda::abs(sobelx, sobelx);
	sobelx.convertTo(output_image, CV_8UC1);
}