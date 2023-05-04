cmake_minimum_required(VERSION 3.5)
project(detect_test)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_PREFIX_PATH /home/chenzhen/Openvino/openvino/build)

include_directories(
        #OpenVINO推理引擎的头文件
        /home/chenzhen/Openvino/openvino/src/inference/include/ie
        /home/chenzhen/Openvino/openvino/src/core/include/ngraph
        # Eigen
        /usr/include/eigen3

)


#查找必要的依赖包
find_package(OpenCV REQUIRED)
set(InferenceEngine_DIR "/home/chenzhen/Openvino/openvino/src/inference/include/ie")
find_package(InferenceEngine)
set(ngraph_DIR "/home/chenzhen/Openvino/openvino/src/core/include/ngraph")
find_package(ngraph REQUIRED)
set(ngraph_LIBRARIES "/home/chenzhen/Openvino/openvino/bin/intel64/Release/libngraph_reference.a")
set(ngraph_INCLUDE_DIRS "/home/chenzhen/Openvino/openvino/src/core/include/ngraph")
set(eigen_LIBRARIES "/usr/include/eigen3")


# 编译detector.so
add_library (detector SHARED ${CMAKE_CURRENT_SOURCE_DIR}/detector.cpp)
target_include_directories(detector
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
        PUBLIC ${eigen_LIBRARIES}
        #        PUBLIC ${ngraph_INCLUDE_DIRS}
        )
target_link_libraries(detector
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
        )

add_executable(detect_test main.cpp)
target_link_libraries(detect_test
        ${OpenCV_LIBS}
        detector
        )