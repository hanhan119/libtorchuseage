cmake_minimum_required(VERSION 2.8,3)
project(qqq_test)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

set(CMAKE_MODULE_PATH 
					#${CMAKE_MODULE_PATH} 
					"/home/nov/qqq_ws/src/qqq_test/libtorch"
					#"/home/nov/qqq_ws/src/qqq_test/libtorch/share/cmake/Torch"
)


set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
)

find_package(OpenCV REQUIRED)

SET(Torch_DIR "/home/nov/qqq_ws/src/qqq_test/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

catkin_package(
)
include_directories(/home/nov/qqq_ws/src/qqq_test/libtorch/include
					/home/nov/qqq_ws/src/qqq_test/libtorch/include/torch/csrc/api/include)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIR}
)


add_executable(qqq src/main.cpp)
target_link_libraries(qqq
                      ${catkin_LIBRARIES} 
                      ${OpenCV_LIBS}  
                      ${TORCH_LIBRARIES}  
                      #/home/nov/qqq_ws/src/qqq_test/libtorch/lib/libtorch.so
                      
) 
set_property(TARGET qqq PROPERTY CXX_STANDARD 11)
