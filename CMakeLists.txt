cmake_minimum_required(VERSION 2.8.3)
project(stereocl)

#FindOpenCL()
#find_package(v4l2)
find_package(OpenCV 3.0.0 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

include_directories(~/Mali_OpenCL_SDK_v1.1.0/include)

#aux_source_directory(. SRC_LIST)

add_executable(${PROJECT_NAME} main.cpp camera.cpp common.cpp image.cpp
    ocvCalib/stereo_calib.cpp
    ocvCalib/stereo_match.cpp
)

target_link_libraries(${PROJECT_NAME} 
    OpenCL 
    opencv_core
    opencv_imgproc
    opencv_highgui
    opencv_ml
    opencv_video
    opencv_videoio
    opencv_features2d
    opencv_calib3d
    opencv_objdetect
    opencv_contrib
    opencv_legacy
    opencv_flann
)


file(COPY assets DESTINATION ${PROJECT_BINARY_DIR})
