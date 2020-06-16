#include "../../include/allIncludes.h"

void LaneDetector::morphology_filters(cv::cuda::GpuMat& src, cv::cuda::GpuMat dst, cv::cuda::Stream& stream)
{
	imgs.erode->apply(src, imgs.erode_dst);

	cv::Mat dilate_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(imgs.dilate * 2 + 1, imgs.dilate * 2 + 1));
	cv::Ptr<cv::cuda::Filter> dilate = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, src.type(), dilate_element);
	dilate->apply(src, imgs.dilate_dst);
} 