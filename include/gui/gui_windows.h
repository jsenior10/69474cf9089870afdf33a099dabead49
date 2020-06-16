#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudacodec.hpp>
#include <thread>

class gui_windows
{
public:
	gui_windows(std::string window_name);

	gui_windows(std::string window_name, const std::string& filename);

	void set_source(const cv::cuda::GpuMat& source);

	void set_source(const cv::Mat& source);

	void set_source(cv::VideoCapture cap);

	void set_size(int height, int width) const;

	cv::cuda::GpuMat create_blank_source() const;

	void display_image();

	void display_image(const cv::cuda::GpuMat& gpu_frame);

private:
	cv::cuda::GpuMat source;

	int height, width;

	int font_face = cv::FONT_HERSHEY_SIMPLEX;

	double font_scale = 2;

	int thickness = 3;

	int baseline = 0;

	std::string window_name;
};
