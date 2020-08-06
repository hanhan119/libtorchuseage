/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "tic_toc.h"


#define HEIGHT  480
#define WIDTH  768

using namespace std;


cv::Mat Classfier(cv::Mat &image,torch::jit::script::Module module)
{
	TicToc a;
	a.tic();

	cv::Mat image_cv_mat = image.clone();
	cv::resize(image_cv_mat,image_cv_mat, cv::Size(WIDTH,HEIGHT));
  if(image_cv_mat.rows != HEIGHT || image_cv_mat.cols != WIDTH)   //判断图片大小
  {
		std::cout<<"something wrong with input image"<<std::endl;
    std::cout << "image.rows=" << image_cv_mat.rows << "    image.cols=" << image_cv_mat.cols << std::endl;
    cv::Mat zerosimg;
    return zerosimg;
  }
  torch::Tensor image_tensor = torch::from_blob(image_cv_mat.data, {1, image_cv_mat.rows, image_cv_mat.cols, 3}, torch::kByte);
	std::cout<<"image_tensor.dim()"<<image_tensor.dim()<<endl;
  image_tensor = image_tensor.permute({0, 3, 1, 2});
  image_tensor = image_tensor.toType(torch::kFloat);
  image_tensor[0][0] = image_tensor[0][0].sub(torch::Scalar(103.939));
  image_tensor[0][1] = image_tensor[0][1].sub(torch::Scalar(116.779));
  image_tensor[0][2] = image_tensor[0][2].sub(torch::Scalar(123.68));
	image_tensor = image_tensor.cuda();
	std::cout << "前处理 using time  : " << a.toc() << "s" << std::endl;
//   //std::vector<torch::jit::IValue> inputs;

//   //inputs.push_back(image_tensor.to(device));
//   //at::Tensor output = module.forward(inputs).toTensor();
//   //auto output = module.forward(inputs).toTuple()->elements();
  auto output = module.forward(std::vector<torch::jit::IValue>{image_tensor}).toTuple()->elements();
	std::cout << "纯 classifier using time  : " << a.toc() << "s" << std::endl;
  //auto pole_torch_tensor = output[0].toTensor().cpu();
	auto pole_torch_tensor = output[0].toTensor().cpu();
  auto lane_torch_tensor = output[1].toTensor().cpu();
	std::cout<<"image_tensor.dim()"<<pole_torch_tensor.dim()<<endl;
	pole_torch_tensor= pole_torch_tensor.slice(/*dim=*/1, /*start=*/0, /*end=*/5) ;
	std::cout<<"image_tensor.dim()"<<pole_torch_tensor.dim()<<endl;
	std::cout << "图片到cpu  : " << a.toc() << "s" << std::endl;
	cv::Mat zerosimg = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	std::cout << "建一张纯黑图  : " << a.toc() << "s" << std::endl;

	auto pole_out=pole_torch_tensor.accessor<long,2>();
	auto lane_out=lane_torch_tensor.accessor<long,2>();

	//std::cout<<"zerosimg.rows="<<zerosimg.rows<<"       zerosimg.cols="<<zerosimg.cols<<std::endl;
  for (int i = 0; i < WIDTH * HEIGHT; ++i) 
  {
    int y = i / WIDTH;
    int x = i % WIDTH;
    if (pole_out[y][x] > 0)
		{
			//image_cv_mat.at<unsigned char>(3 * i + 1) = 125;
			zerosimg.at<unsigned char>(y,x) = 125;
		}
	
    if (lane_out[y][x] > 0)
		{
			//image_cv_mat.at<unsigned char>(3 * i + 2) = 255;
			zerosimg.at<unsigned char>(y,x) = 225;
		}
  }
	std::cout << "后处理 using time  : " << a.toc() << "s" << std::endl; 

  return zerosimg;

}



int main(int argc, char** argv)
{
	ros::init(argc, argv, "wyh");
	ros::NodeHandle n("~");
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

	ros::Publisher pubLeftImage = n.advertise<sensor_msgs::Image>("/leftImage",1000);
	ros::Publisher pubsegmatwyh = n.advertise<sensor_msgs::Image>("/segmatwyhImage",1000);

	if(argc != 2)
	{
		printf("please intput: rosrun vins kitti_odom_test [config file] [data folder] \n"
			   "for example: rosrun vins kitti_odom_test "
			   "~/catkin_ws/src/VINS-Fusion/config/kitti_odom/kitti_config00-02.yaml "
			   "/media/tony-ws1/disk_D/kitti/odometry/sequences/00/ \n");
		return 1;
	}

	string sequence = argv[1];
	printf("read sequence: %s\n", argv[1]);
	string dataPath = sequence + "/";
	
	// load image list
	std::fstream file;
	file.open((dataPath + "/list.txt").c_str());
	if(!file.is_open()){
	    printf("cannot find file: %s\n", (dataPath + "/list.txt").c_str());
	    ROS_BREAK();
	    return 0;          
	}
	std::string imageTime;	//图片list，图片按时间戳存储
	vector<std::string> imageTimeList;
	while ( getline(file, imageTime))
	{
	    imageTimeList.push_back(imageTime);
	}
	file.close();

	string leftImagePath;
	cv::Mat imLeft;
	TicToc load_module;
	load_module.tic();
	torch::jit::script::Module module;
  try 
  {
    module = torch::jit::load("/home/nov/qqq/model_0713.pt");
  }
  catch (const c10::Error &e) 
  {
    std::cerr << "error loading the model" << std::endl;
		return 1;
  }
	try 
  {
    module.to(at::kCUDA);
  }
  catch (const c10::Error &e) 
  {
    std::cerr << "send model to gpu" << std::endl;
		return 1;
  }
	std::cout << "load module to cuda using time  : " << load_module.toc() << "s" << std::endl;
	for (size_t i = 80; i < imageTimeList.size(); i++)
	{	
		if(ros::ok())
		{
			printf("\nprocess image %d\n", (int)i);
			leftImagePath = dataPath + "/image/" + imageTimeList[i] + ".jpg";
			printf("%s\n", leftImagePath.c_str());
			imLeft = cv::imread(leftImagePath);
			//imLeft = imLeft.rowRange(0, 420);
			TicToc classifier;
			classifier.tic();
			cv::Mat segmatwyh = Classfier(imLeft,module);
			std::cout << "classifier using time  : " << classifier.toc() << "s" << std::endl;

			cv::imwrite("/home/nov/qqq_ws/result/testresult"+to_string(i)+".jpg", segmatwyh);
			cout<<"save /home/nov/qqq_ws/result/testresult"<<i<<".jpg"<<endl;
			sensor_msgs::ImagePtr segmatwyhMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", segmatwyh).toImageMsg();
			sensor_msgs::ImagePtr imLeftMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imLeft).toImageMsg();			
			double imageTime;
			stringstream ss(imageTimeList[i]);
			ss >> imageTime;
			//printf("time: %lf\n", imageTime);
			imLeftMsg->header.stamp = ros::Time(imageTime);
			pubLeftImage.publish(imLeftMsg);		//发布原始图片的话题
			
			segmatwyhMsg->header.stamp = ros::Time(imageTime);
			pubsegmatwyh.publish(segmatwyhMsg);		//发布分割之后图片的话题
		}
		else
		{
			cout<<"ros die##########################################"<<endl;
			break;
		}
			
	}

	return 0;
}
