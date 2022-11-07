#pragma once
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class KeypointsDetector
{
public:
	/// <summary>
	/// 随机采样关键点
	/// </summary>
	/// <param name="cloud">输入点云</param>
	/// <param name="keypoints">输出的关键点</param>
	/// <param name="num">需要的关键点数量</param>
	void computeRandomSampleKeypoints(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		int num);
	
	/// <summary>
	/// 体素质心下采样获取关键点
	/// </summary>
	/// <param name="cloud">输入点云</param>
	/// <param name="keypoints">输出的关键点</param>
	/// <param name="L1">体素尺寸,单位：米</param>
	/// <param name="L2"></param>
	/// <param name="L3"></param>
	void computeVoxelGridKeypoints(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		double leaf_size);

	void computeISSKeypoints(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::Normal>::Ptr normals,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		double model_resolution);

};
