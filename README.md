# Face-detection-Demo
一个我自己写的python demo，目前只有做人脸识别，口罩识别，人脸识别用的是包装了dlib的Face_detection 库，口罩识别是用的tensorflow，运行的话，mask_detector.model需要使用https://github.com/chandrikadeb7/Face-Mask-Detection 这个项目的，或者自己有能力自己整一个也行，我是在jetson nano 上运行的这个项目的，目前因为全是在主线程里跑，没开子线程，优化有些差，后面有空优化把
