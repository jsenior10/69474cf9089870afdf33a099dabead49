#include "../../include/imageProcessing/cameraCalibration.h"
#include "../../include/data/allImages.h"

void calibrate_camera(std::string& calibration_images, cv::Size& chessboard_size, transformStruct& ld_s,bool debug)
{
	std::vector<std::vector < cv::Point3f >> obj_points;
	std::vector<std::vector < cv::Point2f >> img_points;

	int num_squares = chessboard_size.width * chessboard_size.height;
	std::vector<cv::Point3f> obj;
	obj.reserve(num_squares);

	for (int i = 0; i < num_squares; i++)
	{
		obj.emplace_back(i / chessboard_size.width, i % chessboard_size.width, 0.0f);
	}

	cv::Size img_size = cv::Size(0, 0);

	for (auto& img_path : std::filesystem::directory_iterator(calibration_images))
	{
		std::string img_string = img_path.path().string();

		cv::Mat img = cv::imread(img_string);
		cv::cuda::GpuMat gpu_img;
		gpu_img.upload(img);
		cv::cuda::GpuMat gray_img(gpu_img.size().height, gpu_img.size().width, CV_8U, 0.0);
		cv::cuda::cvtColor(gpu_img, gray_img, cv::COLOR_BGR2GRAY);

		if (img_size == cv::Size(0, 0))
		{
			img_size = gpu_img.size();
		}

		std::vector<cv::Point2f> corners;
		corners.reserve(num_squares);
		bool corners_found = cv::findChessboardCorners(gray_img, chessboard_size, corners);

		if (corners_found)
		{
			img_points.emplace_back(corners);
			obj_points.emplace_back(obj);

			if (debug)
			{
				cv::drawChessboardCorners(gpu_img, chessboard_size, corners, corners_found);
				cv::imshow(img_string, gpu_img);
				cv::waitKey(500);
			}
		}
	}
	std::vector<cv::Mat> r_vecs;
	std::vector<cv::Mat> t_vecs;
	cv::calibrateCamera(obj_points, img_points, img_size, ld_s.calibration_matrix, ld_s.distance_coefficients, r_vecs, t_vecs);

	ld_s.calibrated = true;
	cv::Mat ncm;
	cv::initUndistortRectifyMap(ld_s.calibration_matrix, ld_s.distance_coefficients, cv::Mat(), ld_s.calibration_matrix, img_size, CV_32FC1, ld_s.cal_out1, ld_s.cal_out2);
}
