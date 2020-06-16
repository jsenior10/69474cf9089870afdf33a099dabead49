#include "../include/imageProcessing/imageProcessing.h"
#include <opencv2/core/cuda.hpp>

cv::cuda::GpuMat colour_filter(cv::cuda::GpuMat input_image)
{
	cv::Scalar lower_yellow = cv::Scalar(18, 94, 140);
	cv::Scalar upper_yellow = cv::Scalar(48, 255, 255);
	cv::Scalar lower_white = cv::Scalar(0, 0, 200);
	cv::Scalar upper_white = cv::Scalar(255, 255, 255);

	cv::cuda::GpuMat combined_image;
	cv::cuda::GpuMat hsv_image;
	cv::cuda::GpuMat masked_white;
	cv::cuda::GpuMat masked_yellow;
	cv::Mat hsv_image_temp;
	cv::Mat masked_white_temp;
	cv::Mat masked_yellow_temp;

	cv::cuda::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);
	hsv_image.download(hsv_image_temp);
	inRange(hsv_image_temp, lower_white, upper_white, masked_white_temp);
	masked_white.upload(masked_white_temp);
	inRange(hsv_image_temp, lower_yellow, upper_yellow, masked_yellow_temp);
	masked_yellow.upload(masked_yellow_temp);

	bitwise_or(masked_white, masked_yellow, combined_image);

	return combined_image;
}

/*cv::cuda::GpuMat thresholding(cv::cuda::GpuMat input_image)
{
	cv::cuda::GpuMat greyscale_image;
	cv::cuda::GpuMat blurred_image;
	cv::cuda::GpuMat canny_image;
	cv::cuda::GpuMat dilated_image;
	cv::cuda::GpuMat eroded_image;

	const cv::Size kernel_size(5, 5);

	cv::cuda::cvtColor(input_image, greyscale_image, cv::COLOR_BGR2GRAY);

	cv::Ptr<cv::cuda::Filter> gaussian_blur_filter = cv::cuda::createGaussianFilter(input_image.type(), blurred_image.type(), kernel_size, 0);
	gaussian_blur_filter->apply(input_image, blurred_image);

	cv::Ptr<cv::cuda::CannyEdgeDetector> canny_edge_detection = cv::cuda::createCannyEdgeDetector(50, 100);
	canny_edge_detection->detect(blurred_image, canny_image);

	int erosion_dilation_size = 5;
	cv::cuda::GpuMat element = cv::cuda::(cv::MORPH_RECT, cv::Size(2 * erosion_dilation_size + 1, 2 * erosionDilation_size + 1));
	cv::Ptr<cv::cuda::Filter> dilation = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, )
}*/
//void draw_lanes(cv::cuda::GpuMat input_image, )

void initialise_trackbars(int frame_height, int frame_width)
{
	std::array<int,8> initial_vals = {497,479,497,826,642,259,642,1091};
	
	cv::namedWindow("Trackbars");
	cv::resizeWindow("Trackbars", 360, 360);
	cv::createTrackbar("TL Height", "Trackbars", &initial_vals[0], 720);
	cv::createTrackbar("TL Width", "Trackbars", &initial_vals[1], 1280);
	cv::createTrackbar("TR Height", "Trackbars", &initial_vals[2], 720);
	cv::createTrackbar("TR Width", "Trackbars", &initial_vals[3], 1280);
	cv::createTrackbar("BL Height", "Trackbars", &initial_vals[4], 720);
	cv::createTrackbar("BL Width", "Trackbars", &initial_vals[5], 1280);
	cv::createTrackbar("BR Height", "Trackbars", &initial_vals[6], 720);
	cv::createTrackbar("BR Width", "Trackbars", &initial_vals[7], 1280);
}

std::array<cv::Point, 4> get_trackbar_vals()
{
	int tl_height = cv::getTrackbarPos("TL Height", "Trackbars");
	int tl_width = cv::getTrackbarPos("TL Width", "Trackbars");
	int tr_height = cv::getTrackbarPos("TR Height", "Trackbars");
	int tr_width = cv::getTrackbarPos("TR Width", "Trackbars");
	int bl_height = cv::getTrackbarPos("BL Height", "Trackbars");
	int bl_width = cv::getTrackbarPos("BL Width", "Trackbars");
	int br_height = cv::getTrackbarPos("BR Height", "Trackbars");
	int br_width = cv::getTrackbarPos("BR Width", "Trackbars");

	std::array<cv::Point,4> warp_points = { cv::Point(tl_width, tl_height), cv::Point(tr_width, tr_height), cv::Point(bl_width, bl_height), cv::Point(br_width, br_height) };

	return warp_points;
}

void draw_warp_points(cv::Mat& input_image, std::array<cv::Point,4> warp_points)
{

	for(auto&& elem : warp_points)
	{
		cv::circle(input_image, elem, 15, (0, 0, 255), cv::FILLED);

	}
}

void get_image_histogram(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& hist_output, cv::cuda::Stream& stream)
{
	cv::cuda::calcHist(input_image, hist_output, stream);
	//cv::cuda::normalize(hist_output, hist_output,0,0,cv::NORM_MINMAX,0,cv::noArray(), stream);
}