#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include "../allIncludes.h"
#include "../data/matrices.h"

void calibrate_camera(std::string& calibration_images, cv::Size& chessboard_size, transformStruct& ld_s, bool debug);
