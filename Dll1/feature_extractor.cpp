#include "pch.h"
#include "feature_extractor.h"
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/fpfh.h>
#include "cfh_hsv_rate.hpp"
//#include <pcl/features/fpfh_omp.h>
#include <pcl/features/fpfh.h>
//#include <pcl/features/shot.h>
#include <pcl/features/shot_omp.h>


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

void FeatureExtractor::computeCFH_HSV_RATE(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, pcl::PointCloud<pcl::Normal>::Ptr normal, double radius, pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature)
{
	pcl::CFH_HSV_RATE<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> es;
	es.setInputCloud(keypoints);
	es.setSearchSurface(cloud);
	es.setInputNormals(normal);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	es.setSearchMethod(tree);
	es.setRadiusSearch(radius);
	es.compute(*feature);
}

void FeatureExtractor::computeFPFH_CFH_HSV_RATE(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, pcl::PointCloud<pcl::Normal>::Ptr normal, double radius, pcl::PointCloud<pcl::PFHSignature125>::Ptr feature)
{
	//计算FPFH
	//pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> es_s;
	pcl::FPFHEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> es_s;
	es_s.setInputCloud(keypoints);
	es_s.setSearchSurface(cloud);
	es_s.setInputNormals(normal);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_1(new pcl::search::KdTree<pcl::PointXYZRGB>());
	es_s.setSearchMethod(tree_1);
	es_s.setRadiusSearch(radius);
	//es_s.setNumberOfThreads(2);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_33(new pcl::PointCloud<pcl::FPFHSignature33>);
	es_s.compute(*fpfh_33);

	//计算CFH_HSV_RATE
	pcl::CFH_HSV_RATE<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> es_c;
	es_c.setInputCloud(keypoints);
	es_c.setSearchSurface(cloud);
	es_c.setInputNormals(normal);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree_2(new pcl::search::KdTree<pcl::PointXYZRGB>());
	es_c.setSearchMethod(tree_2);
	es_c.setRadiusSearch(radius);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr cfh_33(new pcl::PointCloud<pcl::FPFHSignature33>);
	es_c.compute(*cfh_33);

	//拼接两种特征；
	//为了能使用KdTree，输出格式为pcl::PFHSignature125
	feature->resize(keypoints->size());
	for (size_t i = 0; i < feature->size(); i++)
	{
		for (size_t j = 0; j < 33; j++)
		{
			feature->at(i).histogram[j] = fpfh_33->at(i).histogram[j];
		}
		for (size_t k = 0; k < 33; k++)
		{
			feature->at(i).histogram[33 + k] = cfh_33->at(i).histogram[k];
		}
		for (size_t m = 66; m < 125; m++)
		{
			feature->at(i).histogram[m] = 0;
		}
	}
}
void FeatureExtractor::computeSHOT(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, pcl::PointCloud<pcl::Normal>::Ptr normal, double radius, pcl::PointCloud<pcl::SHOT352>::Ptr feature)
{
	//pcl::SHOTEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> es;
	pcl::SHOTEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT352> es;
	es.setInputCloud(keypoints);
	es.setSearchSurface(cloud);
	es.setInputNormals(normal);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	es.setSearchMethod(tree);
	//es.setNumberOfThreads(8);//cpu核心数量
	es.setRadiusSearch(radius);
	/*note: 此处可以不使用es.setLRFRadius，因为当未设置LRF半径时，会自动将其设置为radiusSearch的搜索半径。
	pcl库中的源码：lrf_estimator->setRadiusSearch((lrf_radius_ > 0 ? lrf_radius_ : search_radius_));*/
	es.compute(*feature);
}

void FeatureExtractor::computeCSHOT(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints, pcl::PointCloud<pcl::Normal>::Ptr normal, double radius, pcl::PointCloud<pcl::SHOT1344>::Ptr feature)
{
	//pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> es;
	pcl::SHOTColorEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> es;
	es.setInputCloud(keypoints);
	es.setSearchSurface(cloud);
	es.setInputNormals(normal);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
	es.setSearchMethod(tree);
	//es.setNumberOfThreads(8);//cpu核心数量
	es.setRadiusSearch(radius);
	es.compute(*feature);
}
