#include "pch.h"
#include "keypoints_detector.h"
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/impl/boundary.hpp>
#include <pcl/filters/random_sample.h>
#include <boost/winapi/time.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#ifndef NumberOfThreads
#define NumberOfThreads 2
#endif // !NumberOfThreads

void KeypointsDetector::computeRandomSampleKeypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, int num)
{
	pcl::RandomSample<pcl::PointXYZRGB> rs;
	rs.setInputCloud(cloud);
	rs.setSeed(GetTickCount());//系统启动以来的嘀嗒时间作为随机种子

	rs.setSample(num);
	rs.filter(*keypoints);
}

void KeypointsDetector::computeVoxelGridKeypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, double leaf_size)
{
	pcl::VoxelGrid<pcl::PointXYZRGB> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);
	sor.filter(*keypoints);
}

void KeypointsDetector::computeISSKeypoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, double model_resolution)
{
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	pcl::ISSKeypoint3D<pcl::PointXYZRGB, pcl::PointXYZRGB> iss_detector;
	iss_detector.setNormals(normals);
	iss_detector.setSearchMethod(tree);
	iss_detector.setSalientRadius(6 * model_resolution);
	iss_detector.setNonMaxRadius(4 * model_resolution);
	iss_detector.setThreshold21(0.8);
	iss_detector.setThreshold32(0.8);
	iss_detector.setMinNeighbors(5);
	iss_detector.setNumberOfThreads(2);
	iss_detector.setInputCloud(cloud);
	iss_detector.compute(*keypoints);
}