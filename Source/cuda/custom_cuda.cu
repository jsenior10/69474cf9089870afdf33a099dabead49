#pragma once
#include "../../include/cuda/custom_cuda.cuh"
#include <thrust/count.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <opencv2/cudaarithm.hpp>
#include "../../include/allIncludes.h"

namespace custom_cuda
{
	__global__ void in_range_kernel(const cv::cuda::PtrStepSz<uchar3> src, cv::cuda::PtrStepSzb dst,
	                                int lbc0, int ubc0, int lbc1, int ubc1, int lbc2, int ubc2)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= src.cols || y >= src.rows) return;

		uchar3 v = src(y, x);
		//if pixel >= threshold value, turn it white, else turn black
		if (v.x >= lbc0 && v.x <= ubc0 && v.y >= lbc1 && v.y <= ubc1 && v.z >= lbc2 && v.z <= ubc2)
			dst(y, x) = 255;
		else
			dst(y, x) = 0;
	}

	void in_range_gpu(cv::cuda::GpuMat& src, cv::Scalar& lowerb, cv::Scalar& upperb,
	                  cv::cuda::GpuMat& dst)
	{
		//Lower number = sharper result, but slower performance
		const int m = 16;
		int numRows = src.rows, numCols = src.cols;
		if (numRows == 0 || numCols == 0) return;
		// Attention! Cols Vs. Rows are reversed
		const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
		const dim3 blockSize(m, m, 1);

		in_range_kernel<<<gridSize, blockSize>>>(src, dst, lowerb[0], upperb[0], lowerb[1], upperb[1],
		                                         lowerb[2], upperb[2]);
	}

	/*__global__ void get_perspective_transform_kernel(const cv::Point2f src[], const cv::Point2f dst[], int solve_method)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		cv::cuda::GpuMat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
		double a[8][8], b[8];
		cv::cuda::GpuMat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

		for(int i = 0; i < 4; ++i)
		{
			a[i][0] = a[i + 4][3] = src[i].x;
			a[i][1] = a[i + 4][4] = src[i].y;
			a[i][2] = a[i + 4][5] = 1;
			a[i][3] = a[i][4] = a[i][5] =
			a[i + 4][0] = a[i + 4][1] = a[i + 4][2] = 0;
			a[i][6] = -src[i].x * dst[i].x;
			a[i][7] = -src[i].y * dst[i].x;
			a[i + 4][6] = -src[i].x * dst[i].y;
			a[i + 4][7] = -src[i].y * dst[i].y;
			b[i] = dst[i].x;
			b[i + 4] = dst[i].y;
		}
	}*/

	struct isNonZeroIndex
	{
		__host__ __device__ bool operator()(const int& idx)
		{
			return (idx != -1);
		}
	};

	__global__ void kernel_find_indices(const uint8_t *input, int width, int height, int step, float *indicesx, float *indicesy)
	{
		const int x = blockIdx.x * blockDim.x + threadIdx.x;
		const int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x < width && y < height)
		{
			const int tidPixel = y * step + x;
			const int tidIndex = y * width + x;

			int value = input[tidPixel];
			if(value)
			{
				float X = float(x);
				float Y = float(y);
				indicesx[tidIndex] = X;
				indicesy[tidIndex] = Y;
			}
			else
			{
				indicesx[tidIndex] = -1;
				indicesy[tidIndex] = -1;
			}
		}
	}

	__global__ void processing_next(float* PointX_n, float* PointY_n, const float margin, float* left_n, float* right_n, const int N_n, float* LPoint_x, float* LPoint_y, float* RPoint_x, float* RPoint_y)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i < N_n)
		{
			float good_left_inds_n = ((PointX_n[i] > (left_n[0] * pow(float(PointY_n[i]), 2) + left_n[1] * PointY_n[i] + left_n[2] - margin)) & (PointX_n[i] < (left_n[0] * (pow(float(PointY_n[i]), 2)) + left_n[1] * PointY_n[i] + left_n[2] + margin)));
			float good_right_inds_n = ((PointX_n[i] > (right_n[0] * pow(float(PointY_n[i]), 2) + right_n[1] * PointY_n[i] + right_n[2] - margin)) & (PointX_n[i] < (right_n[0] * (pow(float(PointY_n[i]), 2)) + right_n[1] * PointY_n[i] + right_n[2] + margin)));

			if (good_left_inds_n != 0)
			{
				LPoint_x[i] = PointX_n[i];
				LPoint_y[i] = PointY_n[i];
			}
			else
			{
				LPoint_x[i] = -1;
				LPoint_y[i] = -1;
			}
			if (good_right_inds_n != 0)
			{
				RPoint_x[i] = PointX_n[i];
				RPoint_y[i] = PointY_n[i];
			}
			else
			{
				RPoint_x[i] = -1;
				RPoint_y[i] = -1;
			}
		}

	}

	void indices_point(cv::cuda::GpuMat& src, thrust::device_vector<float>& outx, thrust::device_vector<float>& outy)
	{
		int Array_Size = cv::cuda::countNonZero(src);
		thrust::device_vector<float>Point_x(src.rows * src.step);
		thrust::device_vector<float>Point_y(src.rows * src.step);
		uint8_t* imgPtr;
		cudaMalloc((void**)&imgPtr, src.rows * src.step);
		cudaMemcpyAsync(imgPtr, src.ptr<uint8_t>(), src.rows * src.step, cudaMemcpyDeviceToDevice);
		dim3 block(16, 16);
		dim3 grid;
		grid.x = (src.cols + block.x - 1) / block.x;
		grid.y = (src.rows + block.y - 1) / block.y;
		kernel_find_indices << <grid, block >> > (imgPtr, int(src.cols), int(src.rows), int(src.step), thrust::raw_pointer_cast(Point_x.data()), thrust::raw_pointer_cast(Point_y.data()));
		cudaDeviceSynchronize();
		thrust::copy_if(Point_x.begin(), Point_x.end(), outx.begin(), isNonZeroIndex());
		thrust::copy_if(Point_y.begin(), Point_y.end(), outy.begin(), isNonZeroIndex());
		cudaFree(imgPtr);

	}

	void get_non_zero_pixels(cv::cuda::GpuMat& src, std::vector<float>& output_hx, std::vector<float>& output_hy)
	{

		int array_size = cv::cuda::countNonZero(src);
		thrust::device_vector<float>Point_X(array_size);
		thrust::device_vector<float>Point_Y(array_size);
		indices_point(src, Point_X, Point_Y);
		output_hx.resize(array_size);
		output_hy.resize(array_size);
		thrust::copy(Point_X.begin(), Point_X.end(), output_hx.begin());
		thrust::copy(Point_Y.begin(), Point_Y.end(), output_hy.begin());
	}

	void get_non_zero_pixels_next(cv::cuda::GpuMat& src, std::vector<float>& Loutput_hx, std::vector<float>& Loutput_hy, std::vector<float>& Routput_hx, std::vector<float>& Routput_hy)
	{

		std::vector<float>polyright_out_n;
		std::vector<float>polyleft_out_n;

		polyright_out_n = last_fit::polyfit_right;
		polyleft_out_n = last_fit::polyfit_left;
		size_t SIZE_T = 3 * sizeof(float);
		float* right_fit_last = (float*)malloc(SIZE_T);
		float* right_fit_last_d;
		cudaMalloc(&right_fit_last_d, SIZE_T);
		float* left_fit_last = (float*)malloc(SIZE_T);
		float* left_fit_last_d;
		cudaMalloc(&left_fit_last_d, SIZE_T);

		right_fit_last[2] = polyright_out_n[0];
		left_fit_last[2] = polyleft_out_n[0];
		right_fit_last[1] = polyright_out_n[1];
		left_fit_last[1] = polyleft_out_n[1];
		right_fit_last[0] = polyright_out_n[2];
		left_fit_last[0] = polyleft_out_n[2];

		cudaMemcpy(right_fit_last_d, right_fit_last, SIZE_T, cudaMemcpyHostToDevice);
		cudaMemcpy(left_fit_last_d, left_fit_last, SIZE_T, cudaMemcpyHostToDevice);


		const float margin = 10;
		const int Size_array = cv::cuda::countNonZero(src);
		thrust::device_vector<float>Point_X(Size_array);
		thrust::device_vector<float>Point_Y(Size_array);
		indices_point(src, Point_X, Point_Y);

		float* arrayx = thrust::raw_pointer_cast(&Point_X[0]);
		float* arrayy = thrust::raw_pointer_cast(&Point_Y[0]);

		thrust::device_vector<float>LPoint_x(Size_array);
		thrust::device_vector<float>LPoint_y(Size_array);
		thrust::device_vector<float>RPoint_x(Size_array);
		thrust::device_vector<float>RPoint_y(Size_array);

		processing_next << <Size_array, 1 >> > (arrayx, arrayy, margin, left_fit_last_d, right_fit_last_d, Size_array, thrust::raw_pointer_cast(LPoint_x.data()),
			thrust::raw_pointer_cast(LPoint_y.data()), thrust::raw_pointer_cast(RPoint_x.data()), thrust::raw_pointer_cast(RPoint_y.data()));
		cudaDeviceSynchronize();

		int nonZeroCountL = int(thrust::count_if(LPoint_x.begin(), LPoint_x.end(), isNonZeroIndex()));
		int nonZeroCountR = int(thrust::count_if(RPoint_x.begin(), RPoint_x.end(), isNonZeroIndex()));

		thrust::device_vector<float>Loutx(nonZeroCountL);
		thrust::copy_if(LPoint_x.begin(), LPoint_x.end(), Loutx.begin(), isNonZeroIndex());
		Loutput_hx.resize(nonZeroCountL);
		thrust::copy(Loutx.begin(), Loutx.end(), Loutput_hx.begin());

		thrust::device_vector<float>Louty(nonZeroCountL);
		thrust::copy_if(LPoint_y.begin(), LPoint_y.end(), Louty.begin(), isNonZeroIndex());
		Loutput_hy.resize(nonZeroCountL);
		thrust::copy(Louty.begin(), Louty.end(), Loutput_hy.begin());

		thrust::device_vector<float>Routx(nonZeroCountR);
		thrust::copy_if(RPoint_x.begin(), RPoint_x.end(), Routx.begin(), isNonZeroIndex());
		Routput_hx.resize(nonZeroCountR);
		thrust::copy(Routx.begin(), Routx.end(), Routput_hx.begin());

		thrust::device_vector<float>Routy(nonZeroCountR);
		thrust::copy_if(RPoint_y.begin(), RPoint_y.end(), Routy.begin(), isNonZeroIndex());
		Routput_hy.resize(nonZeroCountR);
		thrust::copy(Routy.begin(), Routy.end(), Routput_hy.begin());

		cudaFree(right_fit_last_d);
		cudaFree(left_fit_last_d);
	}

}
