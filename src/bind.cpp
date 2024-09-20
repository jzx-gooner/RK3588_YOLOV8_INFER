#include <pybind11/pybind11.h>
#include <opencv2/opencv.hpp>
#include "task/yolov8_custom.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "engine/engine.h"
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

class YoloInfer { 
public:
	YoloInfer(std::string model_file){
         const char* model_file_c = model_file.c_str();
         yolo.LoadModel(model_file_c);
         }

	      std::vector<Detection> commit(const py::array& image){
            if(!image.owndata())
			      throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");
            std::vector<Detection> objects;
            cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
            yolo.Run(cvimage,objects);
            return objects;
	}
private:
      Yolov8Custom yolo;
}; 



PYBIND11_MODULE(sgai_yolo, m){

   //  py::class_<cv::Scalar>(m, "Scalar")
   //     .def(py::init<>())
   //     .def(py::init<float, float, float, float>())
   //     .def_readwrite("val", &cv::Scalar::val);

    py::class_<cv::Rect>(m, "Rect")
       .def(py::init<>())
       .def(py::init<int, int, int, int>())
       .def_readwrite("x", &cv::Rect::x)
       .def_readwrite("y", &cv::Rect::y)
       .def_readwrite("width", &cv::Rect::width)
       .def_readwrite("height", &cv::Rect::height);

    py::class_<Detection>(m, "Detection")
       .def(py::init<>())
       .def_readwrite("class_id", &Detection::class_id)
       .def_readwrite("className", &Detection::className)
       .def_readwrite("confidence", &Detection::confidence)
       .def_readwrite("color", &Detection::color)
       .def_readwrite("box", &Detection::box);

    py::class_<std::vector<Detection>>(m, "VectorDetection")
       .def(pybind11::init<>())
       .def("append", [](std::vector<Detection> &vec, const Detection &d) {
            vec.push_back(d);
        })
       .def("__len__", [](const std::vector<Detection> &vec) {
            return vec.size();
        })
       .def("__getitem__", [](const std::vector<Detection> &vec, size_t index) {
            if (index >= vec.size())
                throw pybind11::index_error();
            return vec[index];
        });


    py::class_<YoloInfer>(m, "Yolo")
		.def(py::init<string>(), py::arg("model_path"))
		.def("commit", &YoloInfer::commit, py::arg("img_array"));

}
