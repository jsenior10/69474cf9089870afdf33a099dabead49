#include "../../include/allIncludes.h"
#include "../../include/cuda/cuda_streams.h"

void LaneDetector::first_frame_processing(cv::cuda::GpuMat& src, std::vector<float>& polyright, std::vector<float>& polyleft)
{
	cv::cuda::GpuMat hist;
	cv::cuda::reduce(src(cv::Rect(0, src.rows / 2, src.cols, src.rows / 2)), hist, 0, cv::REDUCE_SUM, CV_32S, cuda_streams::stream1);
	float margin = 60;
	float minpix = 50;
	float windows_no = 20;
	float src_rows = float(src.rows);
	float windows_height = src_rows / windows_no;
	std::vector<float>main_hx;
	std::vector<float>main_hy;
	custom_cuda::get_non_zero_pixels(src, main_hx, main_hy);
	std::vector<float>leftx;
	std::vector<float>lefty;
	std::vector<float>rightx;
	std::vector<float>righty;
	//cv::reduce(src(cv::Rect(0, src.rows / 2, src.cols, src.rows / 2)), hist, 0, cv::REDUCE_SUM, CV_32S);
	int midpoint = (int(hist.cols / 2));
	cv::Point max_locL, max_locR;
	cv::cuda::minMaxLoc(hist(cv::Rect(50, 0, midpoint, hist.rows)), NULL, NULL, NULL, &max_locL);
	cv::cuda::minMaxLoc(hist(cv::Rect(midpoint, 0, midpoint, hist.rows)), NULL, NULL, NULL, &max_locR);
	float leftxbase = float(int(max_locL.x + 50));
	float rightxbase = float(int(max_locR.x + midpoint));


	for (int window = 1; window <= windows_no; window++)
	{
		std::vector<float>leftx_t;
		std::vector<float>lefty_t;
		std::vector<float>rightx_t;
		std::vector<float>righty_t;
		float win_y_low = float(int(src_rows - (window + 1) * windows_height));
		float win_y_high = float(int(src_rows - window * windows_height));
		float winxleft_low = float(int(leftxbase - margin));
		float winxleft_high = float(int(leftxbase + margin));
		float winxright_low = float(int(rightxbase - margin));
		float winxright_high = float(int(rightxbase + margin));
		float mean_left = 0;
		float mean_right = 0;

		for (auto idx = 0; idx < main_hy.size(); idx++)
		{
			float good_left_inds = float((float(main_hy[idx]) >= win_y_low) & (float(main_hy[idx]) < win_y_high) & (float(main_hx[idx]) >= winxleft_low) & (float(main_hx[idx]) < winxleft_high));
			float good_right_inds = float((float(main_hy[idx]) >= win_y_low) & (float(main_hy[idx]) < win_y_high) & (float(main_hx[idx]) >= winxright_low) & (float(main_hx[idx]) < winxright_high));
			if (good_left_inds != 0.f)
			{
				leftx_t.push_back(float(main_hx[idx]));
				lefty_t.push_back(float(main_hy[idx]));
				mean_left = mean_left + float(main_hx[idx]);
			}
			if (good_right_inds != 0.f)
			{
				rightx_t.push_back(float(main_hx[idx]));
				righty_t.push_back(float(main_hy[idx]));
				mean_right = mean_right + float(main_hx[idx]);
			}

		}
		if (leftx_t.size() > minpix)
		{
			leftxbase = float(int(mean_left / leftx_t.size()));
		}
		if (rightx_t.size() > minpix)
		{
			rightxbase = float(int(mean_right / rightx_t.size()));
		}

		leftx.insert(leftx.end(), leftx_t.begin(), leftx_t.end());
		lefty.insert(lefty.end(), lefty_t.begin(), lefty_t.end());
		rightx.insert(rightx.end(), rightx_t.begin(), rightx_t.end());
		righty.insert(righty.end(), righty_t.begin(), righty_t.end());


	}

	polyright = LaneDetector::polyfit_eigen(righty, rightx, 2);
	polyleft = LaneDetector::polyfit_eigen(lefty, leftx, 2);
}

void LaneDetector::next_frame_processing(cv::cuda::GpuMat& src, std::vector<float>& polyright_n, std::vector<float>& polyleft_n)
{
	std::vector<float>leftx;
	std::vector<float>lefty;
	std::vector<float>rightx;
	std::vector<float>righty;

	custom_cuda::get_non_zero_pixels_next(src, leftx, lefty, rightx, righty);

	LaneDetector poly;
	tbb::tbb_thread th1([&righty, &rightx, &polyright_n, &poly]()
		{

			polyright_n = poly.polyfit_eigen(righty, rightx, 2);
		});

	tbb::tbb_thread th2([&lefty, &leftx, &polyleft_n, &poly]()
		{
			polyleft_n = poly.polyfit_eigen(lefty, leftx, 2);
		});

	th1.join();
	th2.join();
}

std::vector<float>last_fit::polyfit_left;
std::vector<float>last_fit::polyfit_right;


void LaneDetector::first_or_next__frame(cv::cuda::GpuMat& src, std::vector<float>& polyfitleft_out, std::vector<float>& polyfitright_out)
{


	if ((last_fit::polyfit_left.empty()) && (last_fit::polyfit_right.empty()))
	{
		LaneDetector::first_frame_processing(src, polyfitright_out, polyfitleft_out);
		last_fit::polyfit_right = polyfitright_out;
		last_fit::polyfit_left = polyfitleft_out;
	}
	else
	{
		LaneDetector::next_frame_processing(src, polyfitright_out, polyfitleft_out);
	}

}

void LaneDetector::frame_processing1(cv::cuda::GpuMat& src, cv::cuda::GpuMat& resize, cv::cuda::GpuMat& dst)
{
	cv::cuda::GpuMat binary_framea,gray_framea, resize_framea, hsv_framea, birdview_framea, sobel_frameout, gpu_undisort, threshold_frame;
	LaneDetector::perspective_warp(src, birdview_framea, cuda_streams::stream1);
	cuda_streams::stream1.waitForCompletion();
	LaneDetector::sobel_filter(birdview_framea, sobel_frameout);
	//LaneDetector::colour_filter_white(imgs.warped, imgs.cf_white_output, cuda_streams::stream2);
	//LaneDetector::colour_filter_yellow(imgs.warped, imgs.cf_yellow_output, cuda_streams::stream3);
	LaneDetector::convert_img_to_hsv(birdview_framea, hsv_framea);
	LaneDetector::convert_img_to_gray(birdview_framea, gray_framea);
	//cudaDeviceSynchronize();
	LaneDetector::thresholded_frame(gray_framea, binary_framea);
	//cudaDeviceSynchronize();
	//LaneDetector::colour_filter_combined(imgs.cf_white_output, imgs.cf_yellow_output, imgs.cf_combined_output, cuda_streams::stream1);
	cv::cuda::addWeighted(binary_framea, 0.9, sobel_frameout, 0.1, -1,dst);
}

void left_point(std::vector<int>& left_X, std::vector<int>& main_Y, std::vector<cv::Point2i>& Pointleft)
{
	int m = int(main_Y.size());
	for (int r = 0; r < m; r++)
	{
		Pointleft.emplace_back(left_X[r], main_Y[r]);
	}

}

void right_point(std::vector<int>& right_X, std::vector<int>& main_Y, std::vector<cv::Point2i>& Pointright)
{

	int m = int(main_Y.size());
	for (int r = 0; r < m; r = r + 10)
	{

		int c = 359 - r;
		Pointright.emplace_back(right_X[c], main_Y[c]);

	}

}

void LaneDetector::frame_processing2(cv::Mat& frame, cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst)
{
	cv::cuda::GpuMat cuda_frameout;
	cv::cuda::GpuMat temp;
	std::vector<cv::Point2i> nonZeroCoordinates;
	std::vector<float>polyleft_in;
	std::vector<float>polyright_in;
	std::vector<int>Leftx;
	std::vector<int>rightx;
	std::vector<int>main_y;

	cuda_frameout.upload(frame);
	LaneDetector::morphology_filters(cuda_frameout, imgs.dilate_dst, cuda_streams::stream1);
	LaneDetector::first_or_next__frame(imgs.dilate_dst, polyleft_in, polyright_in);
	LaneDetector::check_curve_validity(polyleft_in, polyright_in, Leftx, rightx, main_y);

	cv::Mat maskImage = cv::Mat(frame.size(), CV_8UC3, cv::Scalar(0));
	std::vector<cv::Point2i>PointLeft;
	std::vector<cv::Point2i>PointRight;

	tbb::tbb_thread th1([&rightx, &main_y, &PointRight]()
		{
			right_point(rightx, main_y, PointRight);
		});

	tbb::tbb_thread th2([&Leftx, &main_y, &PointLeft]()
		{
			left_point(Leftx, main_y, PointLeft);
		});

	th1.join();
	th2.join();

	std::vector<cv::Point2i>PointLeftRight;
	PointLeft.insert(PointLeft.end(), PointRight.begin(), PointRight.end());
	PointLeftRight = PointLeft;

	temp.upload(maskImage);
	LaneDetector::inverse_perspective_warp(temp, imgs.unwarped_frame, cuda_streams::stream1);

	cv::cuda::addWeighted(src, 1, imgs.unwarped_frame, 0.5, -1, dst);
}

void LaneDetector::resize_frame(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int height, int width)
{
	cv::cuda::resize(src, dst, cv::Size(width, height));
}

void LaneDetector::convert_img_to_hsv(cv::cuda::GpuMat src, cv::cuda::GpuMat dst)
{
	cv::cuda::GpuMat hsv_frame, temp;
	cv::cuda::GpuMat channels_device[3];
	cv::cuda::GpuMat channels_device_dest[3];
	cv::cuda::cvtColor(src, hsv_frame, cv::COLOR_BGR2HSV);
	cv::cuda::split(hsv_frame, channels_device);
	cv::cuda::threshold(channels_device[0], channels_device_dest[0], 0, 100, cv::THRESH_BINARY);
	cv::cuda::threshold(channels_device[2], channels_device_dest[1], 210, 255, cv::THRESH_BINARY);
	cv::cuda::threshold(channels_device[2], channels_device_dest[2], 200, 255, cv::THRESH_BINARY);
	cv::cuda::merge(channels_device_dest, 3, temp);
	cv::cuda::cvtColor(temp, dst, cv::COLOR_HSV2BGR);
	std::cout << "a";
}

void LaneDetector::convert_img_to_gray(cv::cuda::GpuMat src, cv::cuda::GpuMat dst)
{
	cv::Mat temp1, temp2;
	src.download(temp1);
	//cv::cuda::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	cv::cvtColor(temp1, temp2, cv::COLOR_BGR2GRAY);
	dst.upload(temp2);
}

void LaneDetector::thresholded_frame(cv::cuda::GpuMat src, cv::cuda::GpuMat dst)
{
	cv::Mat temp1, temp2;
	src.download(temp1);
	int threshold_val = 110;
	int max = 255;
	//cv::cuda::threshold(src, dst, threshold_val, max, cv::THRESH_BINARY);
	cv::threshold(temp1, temp2, threshold_val, max, cv::THRESH_BINARY);
	dst.upload(temp2);
}
