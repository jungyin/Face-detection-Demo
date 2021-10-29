# Face-detection-Demo
一个我自己写的python demo，目前只有做人脸识别，口罩识别，人脸识别用的是包装了dlib的Face_detection 库，口罩识别是用的tensorflow，运行的话，mask_detector.model需要使用https://github.com/chandrikadeb7/Face-Mask-Detection 这个项目的，或者自己有能力自己整一个也行，我是在jetson nano 上运行的这个项目的，目前因为全是在主线程里跑，没开子线程，优化有些差，后面有空优化把

需要的第三方：tensorflow，opencv，dlib(jetson安装的时候需要注意一下，不能直接装),face_detector(需要先装dlib)

关于运行：
在接着摄像头的情况下，并且camera.py同级目录下有mask_detector.model时：
python3 camera.py
如果没有摄像头，需要
python3 camera.py -v{你的视频的路径}
如果没有mask_detector.model:
python3 camera.py -m{你的口罩模型路径}
