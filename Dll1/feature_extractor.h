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


	/// <summary>
	/// ����ɫ��ֱ��ͼ��cfh_hsv_rate��
	/// </summary>
	/// <param name="cloud">�������</param>
	/// <param name="keypoints">����ؼ���</param>
	/// <param name="normal">���뷨��</param>
	/// <param name="radius">��������뾶</param>
	/// <param name="feature">�������</param>
	void computeCFH_HSV_RATE(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		pcl::PointCloud<pcl::Normal>::Ptr normal,
		double radius,
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature);

	/// <summary>
	/// ���Σ�FPFH����ɫ�ʣ�cfh_hsv_fpfh_rate����ƴ����������66ά������洢��pfh125ֱ��ͼ
	/// </summary>
	/// <param name="cloud">�������</param>
	/// <param name="keypoints">����ؼ���</param>
	/// <param name="normal">���뷨��</param>
	/// <param name="radius">��������뾶</param>
	/// <param name="feature">�������</param>
	void computeFPFH_CFH_HSV_RATE(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		pcl::PointCloud<pcl::Normal>::Ptr normal,
		double radius,
		pcl::PointCloud<pcl::PFHSignature125>::Ptr feature);

	/// <summary>
	/// ����SHOT����
	/// </summary>
	/// <param name="cloud">�������</param>
	/// <param name="keypoints">����ؼ���</param>
	/// <param name="normal">���뷨��</param>
	/// <param name="radius">��������뾶</param>
	/// <param name="pfh">�������</param>
	void computeSHOT(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		pcl::PointCloud<pcl::Normal>::Ptr normal,
		double radius,
		pcl::PointCloud<pcl::SHOT352>::Ptr feature);

	/// <summary>
	/// ����CSHOT����
	/// </summary>
	/// <param name="cloud">�������</param>
	/// <param name="keypoints">����ؼ���</param>
	/// <param name="normal">���뷨��</param>
	/// <param name="radius">��������뾶</param>
	/// <param name="feature">������� SHOT1344</param>
	void computeCSHOT(
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints,
		pcl::PointCloud<pcl::Normal>::Ptr normal,
		double radius,
		pcl::PointCloud<pcl::SHOT1344>::Ptr feature);
};