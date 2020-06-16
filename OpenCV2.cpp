// OpenCV2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/gui/gui_windows.h"
#include "include/imageProcessing/imageProcessing.h"
#include <opencv2/cudacodec.hpp>
#include <thread>
#include <memory>
#include <time.h>
#include "include/data/allImages.h"

LaneDetector LD;
transformStruct ld_s;
colour_boundaries cb;
all_images imgs;


int main()
{
	cv::cuda::setDevice(0);
	const std::string fname = "project_video (1).mp4";
	const std::string outname = "test.mp4";
	cv::Ptr<cv::cudacodec::VideoReader> d_reader = cv::cudacodec::createVideoReader(fname);
	cv::cuda::Stream stream1, stream2, stream3, stream4;
	cv::Mat cudaout_frame;

	int frame_count = 0;
	initialise_trackbars(720, 1280);
	double seconds, fps;
	time_t start, end;
	std::time(&start);
	LD.initialise_colour_boundaries();
	LD.create_transformation_matrix(imgs.gpu_frame, get_trackbar_vals());
	LD.create_transformation_matrix(imgs.gpu_frame, LD.warp_points);
	LD.create_inv_transformation_matrix(imgs.gpu_frame, LD.warp_points);
	while (true)
	{
		//cap.read(frame);
		if (!d_reader->nextFrame(imgs.gpu_frame))
			break;
		/*LD.colour_filter_white(imgs.gpu_frame, imgs.cf_white_output, stream1);
		LD.colour_filter_yellow(imgs.gpu_frame, imgs.cf_yellow_output, stream2);

		LD.colour_filter_combined(imgs.cf_white_output, imgs.cf_yellow_output, imgs.cf_combined_output, stream1);
		stream1.waitForCompletion();

		LD.perspective_warp(imgs.cf_combined_output, imgs.inv, stream3);
		stream3.waitForCompletion();

		LD.sobel_filter(imgs.gpu_frame, imgs.sobel);

		//LD.get_image_histogram(imgs.inv, imgs.hist_out, stream3);

		cudaDeviceSynchronize();*/
		LD.frame_processing1(imgs.gpu_frame, imgs.resized, imgs.frame_processed1);
		imgs.frame_processed1.download(cudaout_frame);
		LD.resize_frame(imgs.gpu_frame, imgs.resized, 448, 448);
		LD.frame_processing2(cudaout_frame, imgs.resized, imgs.processed_frame);
		frame_count++;
		if (cv::waitKey(1) > 0)
			break;
	}
	std::time(&end);
	seconds = difftime(end, start);
	fps = frame_count / seconds;
	std::cout << fps << " Average FPS\n";
	cv::destroyAllWindows();
	std::cout << "Hello World!\n";
}
