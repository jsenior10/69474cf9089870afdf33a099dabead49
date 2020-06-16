#pragma once
#include <opencv2/cudalegacy/NCVPyramid.hpp>
#include "device_launch_parameters.h"
#include <opencv2/core.hpp>
#include <opencv2/cudacodec.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

namespace custom_cuda
{
	__global__ void in_range_kernel(cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSzb dst,
	                                int lbc0, int ubc0, int lbc1, int ubc1, int lbc2, int ubc2);

	/**
	 * \brief GPU implementation of cv::inRange, lots of speedup, but I'm using a GTX1080 so results may vary
	 * \param src input GpuMat
	 * \param lowerb lower colour boundaries
	 * \param upperb upper colour boundaries
	 * \param dst output GpuMat - ALWAYS INITIALISE GPUMAT SIZE BEFORE RUNNING THIS FUNCTION OR YOU WILL GET MEMORY EXCEPTION
	 */
	void in_range_gpu(cv::cuda::GpuMat& src, cv::Scalar& lowerb, cv::Scalar& upperb,
	                  cv::cuda::GpuMat& dst);

	void get_non_zero_pixels(cv::cuda::GpuMat& src, std::vector<float>& output_hx, std::vector<float>& output_hy);
	void get_non_zero_pixels_next(cv::cuda::GpuMat& src, std::vector<float>& Loutput_hx, std::vector<float>& Loutput_hy, std::vector<float>& Routput_hx, std::vector<float>& Routput_hy);
}
