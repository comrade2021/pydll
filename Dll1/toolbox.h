#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>

class ToolBox
{
public:
	//计算所给点云的网格分辨率（根据近邻点计算）
	float computeMeshResolution(pcl::PointCloud<pcl::PointXYZRGB>::Ptr input);
	//计算点云分辨率
	double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& cloud);

};