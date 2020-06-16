#pragma once
#include "../allIncludes.h"
#include "../cuda/custom_cuda.cuh"


cv::cuda::GpuMat denoise_image(cv::cuda::GpuMat input_image);
void initialise_colour_boundaries();
bool undistort(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::Mat& cal_out1, cv::Mat& cal_out2, bool& calibrated);
void calibrate_camera(std::string& calibration_images, cv::Size& chessboard_size, bool& calibrated, cv::cuda::GpuMat& calibration_matrix, cv::cuda::GpuMat& distance_coefficients, cv::Mat& cal_out1, cv::Mat& cal_out2, bool debug = false);
void perspective_warp(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
void create_inv_transformation_matrix(cv::cuda::GpuMat& input_image, std::array<cv::Point, 4> warp_points);
void inverse_perspective_warp(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
cv::cuda::GpuMat colour_filter(cv::cuda::GpuMat input_image);
void colour_filter_white(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
void colour_filter_yellow(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
cv::cuda::GpuMat colour_filter(const cv::cuda::GpuMat input_image, cv::cuda::GpuMat output_image);
void colour_filter_combined(cv::cuda::GpuMat& masked_white, cv::cuda::GpuMat& masked_yellow, cv::cuda::GpuMat& output_image, cv::cuda::Stream& stream);
void sobel_filter(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image);
cv::cuda::GpuMat detect_edges(cv::cuda::GpuMat denoised_image);
void create_transformation_matrix(cv::cuda::GpuMat& input_image, std::array<cv::Point, 4> warp_points);
cv::cuda::GpuMat mask(cv::cuda::GpuMat edge_image);
cv::cuda::GpuMat thresholding(cv::cuda::GpuMat input_image);
std::vector<cv::Vec4i> hough_lines(cv::cuda::GpuMat mask_image);
std::vector<std::vector<cv::Vec4i>> line_separation(std::vector<cv::Vec4i> lines, cv::cuda::GpuMat edge_image);
std::vector<cv::Point> regression(std::vector<std::vector<cv::Vec4i>> left_right_lines, cv::cuda::GpuMat input_image);
std::string predict_turn();
void get_image_histogram(cv::cuda::GpuMat input_image, cv::cuda::GpuMat& hist_output, cv::cuda::Stream& stream);
int plot_lane(cv::cuda::GpuMat input_image, std::vector<cv::Point> lane, std::string turn);
void initialise_trackbars(int frame_height, int frame_width);
std::array<cv::Point, 4> get_trackbar_vals();
void draw_warp_points(cv::Mat& input_image, std::array<cv::Point, 4> warp_points);
/**
 * \brief OpenCV doesn't contain a CUDA function for cv::inRange so this is a custom implementation using existing OpenCV functions like cv::cuda::threshold
 * \param src first input GpuMat array, assumes input is already a GpuMat, overload function if you want to use cv::Mat
 * \param lowerb inclusive lower boundary array or scalar
 * \param upperb inclusive upper boundary array or scalar
 * \param dst output GpuMat array
 * \return returns GpuMat array
 */
cv::cuda::GpuMat in_range_gpu(cv::cuda::GpuMat &src, cv::InputArray &lowerb, cv::InputArray &upperb, cv::cuda::GpuMat &dst);
