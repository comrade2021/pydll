#pragma once
#include "pch.h"  
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class FeatureExtractor
{
public:
	/// <summary>
	/// ����FPFH����
	/// </summary>
	/// <param name="cloud">�������</param>
	/// <param name="keypoints">����ؼ���</param>
	/// <param name="normal">���뷨��</param>
	/// <param name="radius">��������뾶</param>
	/// <param name="feature">������� FPFHSignature33</param>
	void computeFPFH(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		pcl::PointCloud<pcl::Normal>::Ptr normal,
		double radius,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature);

};