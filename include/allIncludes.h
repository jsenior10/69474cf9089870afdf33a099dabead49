#pragma once
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudalegacy.hpp>
#include <climits>
#include "cuda/custom_cuda.cuh"
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/cudawarping.hpp>
#include "data/matrices.h"
#include "data/colourBoundaries.h"
#include "data/allImages.h"
#include <tbb/tbb.h>

extern transformStruct ld_s;
extern colour_boundaries cb;
extern all_images imgs;

class LaneDetector
{
public:
	std::array<int, 8> initial_warp_vals = { 497,479,497,826,642,259,642,1091 };
	std::array<cv::Point, 4> warp_points = { cv::Point(initial_warp_vals[1], initial_warp_vals[0]), cv::Point(initial_warp_vals[3], initial_warp_vals[2]), cv::Point(initial_warp_vals[5], initial_warp_vals[4]), cv::Point(initial_warp_vals[7], initial_warp_vals[6]) };
	float centre_dist;
	float left_curve_radians;
	float right_curve_radians;

	void initialise_colour_boundaries();

	std::vector<float> polyfit_eigen(const std::vector<float>& xv, const std::vector<float>& yv, int order);
	std::vector<float> polyvaleigen(const std::vector<float>& oCoeff, const std::vector<float>& oX);

	bool undistort(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::Mat& cal_out1, cv::Mat& cal_out2, bool& calibrated);

	void colour_filter_white(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
	void colour_filter_yellow(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
	void colour_filter_combined(cv::cuda::GpuMat& masked_white, cv::cuda::GpuMat& masked_yellow, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
	void sobel_filter(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image);
	void convert_img_to_gray(cv::cuda::GpuMat src, cv::cuda::GpuMat dst);
	void create_transformation_matrix(cv::cuda::GpuMat& input_image, std::array<cv::Point, 4> warp_points);
	void create_inv_transformation_matrix(cv::cuda::GpuMat& input_image, std::array<cv::Point, 4> warp_points);
	void perspective_warp(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
	void inverse_perspective_warp(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
	void morphology_filters(cv::cuda::GpuMat& src, cv::cuda::GpuMat dst, cv::cuda::Stream& stream);
	void check_curve_validity(std::vector<float>& polyleft_in, std::vector<float>& polyright_in, std::vector<int>& Leftx, std::vector<int>& rightx, std::vector<int>& main_y);
	void thresholded_frame(cv::cuda::GpuMat src, cv::cuda::GpuMat dst);
	void first_or_next__frame(cv::cuda::GpuMat& src, std::vector<float>& polyleft_out, std::vector<float>& polyright_out);
	void first_frame_processing(cv::cuda::GpuMat& src, std::vector<float>& polyright, std::vector<float>& polyleft);
	void next_frame_processing(cv::cuda::GpuMat& src, std::vector<float>& polyright_n, std::vector<float>& polyleft_n);
	void frame_processing1(cv::cuda::GpuMat& src, cv::cuda::GpuMat& resize, cv::cuda::GpuMat& dst);
	void frame_processing2(cv::Mat& frame, cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
	void resize_frame(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int height, int width);
	void convert_img_to_hsv(cv::cuda::GpuMat src, cv::cuda::GpuMat dst);

};

class last_fit
{
public:
	static std::vector<float> polyfit_right;
	static std::vector<float> polyfit_left;
};
