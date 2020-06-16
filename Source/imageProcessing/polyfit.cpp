#include "../../include/allIncludes.h"
#include "../../include/Eigen/QR"
#include "../../include/Eigen/Core"
#include "../../include/cuda/custom_cuda.cuh"
#include <tbb/tbb.h>

std::vector<float> LaneDetector::polyfit_eigen(const std::vector<float> &xv, const std::vector<float>& yv, int order)
{
	Eigen::initParallel();
	Eigen::MatrixXf A = Eigen::MatrixXf::Ones(xv.size(), order + 1);
	Eigen::VectorXf yv_mapped = Eigen::VectorXf::Map(&yv.front(), yv.size());
	Eigen::VectorXf xv_mapped = Eigen::VectorXf::Map(&xv.front(), xv.size());
	Eigen::VectorXf result;

	assert(xv.size() == yv.size());
	assert(xv.size() >= order + 1);

	for (int j = 1; j < order + 1; j++)
	{
		A.col(j) = A.col(j - 1).cwiseProduct(xv_mapped);
	}

	result = A.householderQr().solve(yv_mapped);
	std::vector<float> coeff;
	coeff.resize(order + 1);
	for (size_t i = 0; i < order + 1; i++)
		coeff[i] = result[i];

	return coeff;
}

std::vector<float>LaneDetector::polyvaleigen(const std::vector<float>& oCoeff, const std::vector<float>& oX)
{
	int nCount = int(oX.size());
	int nDegree = int(oCoeff.size());
	std::vector<float>oY(nCount);

	for (int i = 0; i < nCount; i++)
	{
		float nY = 0;
		float nXT = 1;
		float nX = oX[i];
		for (int j = 0; j < nDegree; j++)
		{
			nY += oCoeff[j] * nXT;
			nXT *= nX;
		}
		oY[i] = nY;

	}

	return oY;
}