#pragma once
#include "pch.h"  
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class FeatureExtractor
{
public:
	/// <summary>
	/// 计算FPFH特征
	/// </summary>
	/// <param name="cloud">输入点云</param>
	/// <param name="keypoints">输入关键点</param>
	/// <param name="normal">输入法线</param>
	/// <param name="radius">特征计算半径</param>
	/// <param name="feature">输出特征 FPFHSignature33</param>
	void computeFPFH(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		pcl::PointCloud<pcl::Normal>::Ptr normal,
		double radius,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature);

};