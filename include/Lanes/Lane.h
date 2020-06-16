#pragma once
#include <vector>
#include <opencv2/core/async.hpp>

class Lane
{
private:
	const int num_windows = 6;
	const int window_margin = 100;
	const int min_number_pixels = 200;

	std::vector<cv::Point> sliding_window(cv::cuda::GpuMat& image, std::vector<cv::Point> nonzero_pixels, cv::Point current_base);

	std::vector<double> polyfit(std::vector<cv::Point> points, int degree = 2);

public:
	std::vector<double> find_lane(cv::cuda::GpuMat& image, std::vector<cv::Point> nonzero, cv::Point current_base);
};
