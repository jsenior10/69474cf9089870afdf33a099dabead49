#include <utility>
#include <opencv2/cudacodec.hpp>
#include "../../include/gui/gui_windows.h"

gui_windows::gui_windows(std::string window_name, const std::string& filename)
{
	this->window_name = std::move(window_name);
	cv::namedWindow(this->window_name, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);
	cv::VideoCapture cap(filename);

	if (this->source.empty())
	{
		do
		{
			this->set_source(this->create_blank_source());
			cv::Mat temp;
			this->source.download(temp);
			std::string text = "no image found";
			cv::Size text_size = cv::getTextSize(text, this->font_face, this->font_scale, this->thickness,
				&this->baseline);
			this->baseline += this->thickness;
			cv::Point text_org((this->source.cols - text_size.width) / 2,
				(this->source.rows - text_size.height) / 2);
			putText(temp, text, text_org, this->font_face, this->font_scale, cv::Scalar::all(255), this->thickness, 8);

			this->source.upload(temp);
			this->display_image();
			//temp.release();
		} while (this->source.empty());
	}
	else
	{
		this->display_image();
	}
};

gui_windows::gui_windows(std::string window_name)
{
	this->window_name = std::move(window_name);
	cv::namedWindow(this->window_name, cv::WINDOW_AUTOSIZE | cv::WINDOW_OPENGL);

	if (this->source.empty())
	{
		do
		{
			this->set_source(this->create_blank_source());
			cv::Mat temp;
			this->source.download(temp);
			std::string text = "no image found";
			cv::Size text_size = cv::getTextSize(text, this->font_face, this->font_scale, this->thickness,
			                                     &this->baseline);
			this->baseline += this->thickness;
			cv::Point text_org((this->source.cols - text_size.width) / 2,
			                   (this->source.rows - text_size.height) / 2);
			putText(temp, text, text_org, this->font_face, this->font_scale, cv::Scalar::all(255), this->thickness, 8);

			this->source.upload(temp);
			this->display_image();
			//temp.release();
		}
		while (this->source.empty());
	}
	else
	{
		this->display_image();
	}
}

cv::cuda::GpuMat gui_windows::create_blank_source() const
{
	cv::Mat pic = cv::Mat::zeros(480, 480, CV_8UC3);
	cv::cuda::GpuMat blank;
	blank.upload(pic);
	//pic.release();
	return blank;
}

void gui_windows::set_source(const cv::cuda::GpuMat& source)
{
	this->source = source;
	//source.release();
}

void gui_windows::set_source(const cv::Mat& source)
{
	this->source.upload(source);
	//source.release();
}

void gui_windows::set_source(cv::VideoCapture cap)
{
	cv::Mat frame;
	cap >> frame;
	this->set_source(frame);
	//frame.release();
}

void gui_windows::display_image()
{
	cv::Mat out(this->source);
	imshow(this->window_name, out);
	//out.release();
}

void gui_windows::display_image(const cv::cuda::GpuMat& gpu_frame)
{
	cv::imshow(this->window_name, gpu_frame);
	//out.release();
}

void gui_windows::set_size(int height, int width) const
{
	cv::resizeWindow(this->window_name, height, width);
}