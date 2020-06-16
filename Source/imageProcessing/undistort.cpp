#include "../../include/allIncludes.h"

bool LaneDetector::undistort(cv::cuda::GpuMat& input_image, cv::cuda::GpuMat& output_image, cv::Mat& cal_out1, cv::Mat& cal_out2, bool& calibrated)
{
	if (ld_s.calibrated)
	{
		cv::cuda::remap(input_image, output_image, cal_out1, cal_out2, cv::INTER_CUBIC);
		return true;
	}
	return false;
}