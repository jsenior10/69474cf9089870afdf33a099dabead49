#include "../../include/allIncludes.h"

void initialise_images_struct()
{
	imgs.gpu_frame = cv::cuda::GpuMat(720, 1280, CV_8UC3);
	imgs.cf_white_output = cv::cuda::GpuMat(720, 1280, CV_8UC1);
	imgs.cf_yellow_output = cv::cuda::GpuMat(720, 1280, CV_8UC1);
	imgs.cf_combined_output = cv::cuda::GpuMat(720, 1280, CV_8UC1);
	imgs.hist_out = cv::cuda::GpuMat(1, 256, CV_8UC1);
	imgs.cf_output = cv::cuda::GpuMat(720, 1280, CV_8UC1);
	imgs.inv = cv::cuda::GpuMat(720, 1280, CV_8U);
	imgs.sobel = cv::cuda::GpuMat(720, 1280, CV_64F);

	int noise = 3;
	int dilate_const = 1;
	cv::Mat erosion_element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(imgs.erode_noise * 2 + 1, imgs.erode_noise * 2 + 1));
	cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, imgs.sobel.type(), erosion_element);
}