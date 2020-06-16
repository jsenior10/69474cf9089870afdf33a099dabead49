#include "../../include/allIncludes.h"

void LaneDetector::create_transformation_matrix(cv::cuda::GpuMat& input_image, std::array<cv::Point, 4> warp_points)
{
	cv::Point2f src_p[4];
	cv::Point2f dst_p[4];

	float w = (float)input_image.cols;
	float h = (float)input_image.rows;
	float hw = w / 2.0f;
	float hh = h / 2.0f;

	src_p[0] = cv::Point(warp_points[0].x, warp_points[0].y);
	src_p[1] = cv::Point(warp_points[1].x, warp_points[1].y);
	src_p[3] = cv::Point(warp_points[2].x, warp_points[2].y);
	src_p[2] = cv::Point(warp_points[3].x, warp_points[3].y);

	dst_p[0] = cv::Point(0, 0);		//top left corner
	dst_p[1] = cv::Point(input_image.cols, 0);		//top right corner
	dst_p[2] = cv::Point(input_image.cols, input_image.rows);	//bottom right corner
	dst_p[3] = cv::Point(0, input_image.rows);		//bottom left

	ld_s.transform_matrix = cv::getPerspectiveTransform(src_p, dst_p); //CV_64F->double
}

void LaneDetector::create_inv_transformation_matrix(cv::cuda::GpuMat& input_image, std::array<cv::Point, 4> warp_points)
{
	cv::Point2f src_p[4];
	cv::Point2f dst_p[4];

	float w = (float)input_image.cols;
	float h = (float)input_image.rows;
	float hw = w / 2.0f;
	float hh = h / 2.0f;

	src_p[0] = cv::Point(warp_points[0].x, warp_points[0].y);
	src_p[1] = cv::Point(warp_points[1].x, warp_points[1].y);
	src_p[3] = cv::Point(warp_points[2].x, warp_points[2].y);
	src_p[2] = cv::Point(warp_points[3].x, warp_points[3].y);

	dst_p[0] = cv::Point(0, 0);		//top left corner
	dst_p[1] = cv::Point(1280, 0);		//top right corner
	dst_p[2] = cv::Point(1280, 720);	//bottom right corner
	dst_p[3] = cv::Point(0, 720);		//bottom left

	cv::cuda::GpuMat dst(720, 1280, CV_8UC1);

	ld_s.inv_transform_matrix = cv::getPerspectiveTransform(dst_p, src_p); //CV_64F->double
}

void LaneDetector::perspective_warp(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream)
{
	cv::cuda::warpPerspective(input_image, output_image, ld_s.transform_matrix, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), stream);
}

void LaneDetector::inverse_perspective_warp(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream)
{
	cv::cuda::warpPerspective(input_image, output_image, ld_s.inv_transform_matrix, input_image.size(), cv::INTER_LINEAR);
}