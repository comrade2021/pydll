#include "pch.h"
#include "feature_extractor.h"
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/fpfh.h>


void FeatureExtractor::computeFPFH(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, pcl::PointCloud<pcl::Normal>::Ptr normal, double radius, pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature)
{
	//pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> es;
	pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> es;
	es.setInputCloud(keypoints);
	es.setSearchSurface(cloud);
	es.setInputNormals(normal);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	es.setSearchMethod(tree);
	es.setRadiusSearch(radius);
	//es.setNumberOfThreads(2);
	es.compute(*feature);
}