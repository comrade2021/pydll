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
MATHLIBRARY_API float * compute_keypoints_and_features_fpfh(wchar_t* path)
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

    return result;
}

MATHLIBRARY_API int compute_keypoints_and_features_hsv_rate(wchar_t* path)
{
    int k = 30;//将要采集的关键点数量（仅对随机关键点有效）

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

    pcl::PointCloud<pcl::PFHSignature125>::Ptr cloud_features(new pcl::PointCloud<pcl::PFHSignature125>());
    FeatureExtractor fex;
    fex.computeFPFH_CFH_HSV_RATE(cloud, cloud_keypoints, cloud_normals, feature_radius, cloud_features);
    //导出keypoint和features的numpy文件,先存入vector，再导出为numpy文件
    //库的来源：https://github.com/llohse/libnpy/blob/master/tests/test-save.cpp
    std::vector<float> data_keypoints;
    std::vector<float> data_features;
    for (int i = 0; i < cloud_keypoints->size(); i++) {
        data_keypoints.push_back(cloud_keypoints->at(i).x);
        data_keypoints.push_back(cloud_keypoints->at(i).y);
        data_keypoints.push_back(cloud_keypoints->at(i).z);
    }
    for (int i = 0; i < cloud_features->size(); i++) {
        for (int j = 0; j < 66; j++) {
            data_features.push_back(cloud_features->at(i).histogram[j]);
        }
    }
    std::array<long unsigned, 2> shape1{ cloud_keypoints->size(), 3};
    std::array<long unsigned, 2> shape2{ cloud_features->size(), 66};
    npy::SaveArrayAsNumpy("temp_data/model_keypoints.npy", false, shape1.size(), shape1.data(), data_keypoints);
    npy::SaveArrayAsNumpy("temp_data/model_features.npy", false, shape2.size(), shape2.data(), data_features);

    return cloud_keypoints->size();
}

MATHLIBRARY_API int compute_keypoints_and_features_shot(wchar_t* path)
{
    int k = 30;//将要采集的关键点数量（仅对随机关键点有效）

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

    pcl::PointCloud<pcl::SHOT352>::Ptr cloud_features(new pcl::PointCloud<pcl::SHOT352>());
    FeatureExtractor fex;
    fex.computeSHOT(cloud, cloud_keypoints, cloud_normals, feature_radius, cloud_features);
    //导出keypoint和features的numpy文件,先存入vector，再导出为numpy文件
    //库的来源：https://github.com/llohse/libnpy/blob/master/tests/test-save.cpp
    std::vector<float> data_keypoints;
    std::vector<float> data_features;
    for (int i = 0; i < cloud_keypoints->size(); i++) {
        data_keypoints.push_back(cloud_keypoints->at(i).x);
        data_keypoints.push_back(cloud_keypoints->at(i).y);
        data_keypoints.push_back(cloud_keypoints->at(i).z);
    }
    for (int i = 0; i < cloud_features->size(); i++) {
        for (int j = 0; j < 352; j++) {
            data_features.push_back(cloud_features->at(i).descriptor[j]);
        }
    }
    std::array<long unsigned, 2> shape1{ cloud_keypoints->size(), 3 };
    std::array<long unsigned, 2> shape2{ cloud_features->size(), 352 };
    npy::SaveArrayAsNumpy("temp_data/model_keypoints.npy", false, shape1.size(), shape1.data(), data_keypoints);
    npy::SaveArrayAsNumpy("temp_data/model_features.npy", false, shape2.size(), shape2.data(), data_features);

    return cloud_keypoints->size();
}

MATHLIBRARY_API int compute_keypoints_and_features_cshot(wchar_t* path)
{
    int k = 30;//将要采集的关键点数量（仅对随机关键点有效）

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

    pcl::PointCloud<pcl::SHOT1344>::Ptr cloud_features(new pcl::PointCloud<pcl::SHOT1344>());
    FeatureExtractor fex;
    fex.computeCSHOT(cloud, cloud_keypoints, cloud_normals, feature_radius, cloud_features);
    //导出keypoint和features的numpy文件,先存入vector，再导出为numpy文件
    //库的来源：https://github.com/llohse/libnpy/blob/master/tests/test-save.cpp
    std::vector<float> data_keypoints;
    std::vector<float> data_features;
    for (int i = 0; i < cloud_keypoints->size(); i++) {
        data_keypoints.push_back(cloud_keypoints->at(i).x);
        data_keypoints.push_back(cloud_keypoints->at(i).y);
        data_keypoints.push_back(cloud_keypoints->at(i).z);
    }
    for (int i = 0; i < cloud_features->size(); i++) {
        for (int j = 0; j < 1344; j++) {
            data_features.push_back(cloud_features->at(i).descriptor[j]);
        }
    }
    std::array<long unsigned, 2> shape1{ cloud_keypoints->size(), 3 };
    std::array<long unsigned, 2> shape2{ cloud_features->size(), 1344 };
    npy::SaveArrayAsNumpy("temp_data/model_keypoints.npy", false, shape1.size(), shape1.data(), data_keypoints);
    npy::SaveArrayAsNumpy("temp_data/model_features.npy", false, shape2.size(), shape2.data(), data_features);

    return cloud_keypoints->size();
}