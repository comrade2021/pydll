#include "pch.h"      //导入该文件是编译提醒，不加会报错
#include<iostream> 
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "keypoints_detector.h"
#include "my_normal_estimation.h"
#include "feature_extractor.h"
#include "toolbox.h"

//for numcpp
#include <npy.hpp>

#define MATHLIBRARY_API extern "C" __declspec(dllexport)   //主要就是加入这个宏定义


/// <summary>
/// 读取点云文件，计算并返回关键点坐标和FPFH特征，关键点数量固定为30个
/// </summary>
/// <param name="path">点云路径</param>
/// <returns>result为一个数组，前边存放关键点坐标，后边存放FPFH特征</returns>
MATHLIBRARY_API float * compute_keypoints_and_features(wchar_t* path)
{
    int k = 30;     //将要采集的关键点数量为30
    static float result[30*36]={0};    // 返回的数组必须是全局变量或者静态变量（ctypes）
    
    //获取点云文件路径，wchar转为string
    std::wstring ws(path);
    std::string str_path(ws.begin(), ws.end());

    //点云读取
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(str_path, *cloud) == -1) {
        PCL_ERROR("Couldn't read pcd file\n");
    }

    //点云分辨率或密度计算
    ToolBox tb;
    double mr = tb.computeMeshResolution(cloud);
    //double res = tb.computeCloudResolution(cloud);


    //法线计算
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    MyNormalEstimation ne;
    ne.computeNormal_K(cloud, 10, cloud_normals);//小物体的法线计算


    //关键点检测（随机采样）
    KeypointsDetector kdt;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>);
    kdt.computeRandomSampleKeypoints(cloud, cloud_keypoints, k);

    //特征计算
    double feature_radius = 0.0016 * 6;
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr cloud_features(new pcl::PointCloud<pcl::FPFHSignature33>());
    FeatureExtractor fex;
    fex.computeFPFH(cloud, cloud_keypoints, cloud_normals, feature_radius, cloud_features);

    //返回30个关键点的坐标及特征直方图
    for (int i = 0; i < 30; i++) {
		result[3 * i] = cloud_keypoints->points[i].x;
		result[3 * i + 1] = cloud_keypoints->points[i].y;
		result[3 * i + 2] = cloud_keypoints->points[i].z;
    }

	for (int i = 0; i < 30; i++) {  //i是点的序号，j是直方图序号
		for (int j = 0; j < 33; j++) {
            result[90 + (33 * i) + j] = cloud_features->at(i).histogram[j];
		}
	}

    const std::vector<long unsigned> shape{ 2, 3 };
    const bool fortran_order{ false };
    const std::string path2{ "out.npy" };

    const std::vector<double> data1{ 1, 2, 3, 4, 5, 6 };
    npy::SaveArrayAsNumpy(path2, fortran_order, shape.size(), shape.data(), data1);

    return result;
}