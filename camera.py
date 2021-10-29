import cv2
import face_recognition
import time
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import argparse
# 判断当前是否正在存储照片
saveFace = False
# 图片缩放为原来的0.几
PictureZoom = 0.25
# 读取图片存储的路径
ImagePath = "./facepath"
# 摄像机宽高
CameraHeight = 720
CameraWidth = 1280

# 口罩佩戴阈值
MaskConfidence = 0.6

Debug = True

# 已知人脸的人脸特征数组
known_face_encodings = []
# 已知的人脸的名字
known_face_names = []

# 初始化缓存
process_this_frame = 0


# 视频的存放路径，如果为空，就调用摄像头
videoPath = ""


# 将img2覆盖到img上,通过dxdy来调整所覆盖的位置
def drawableImgToImg(img, img2, dx=0, dy=0, needAlpha=False):
    rows, cols, channels = img2.shape

    if needAlpha:
        roi = img[dy:dy + rows, dx:dx + cols]
        for i in range(cols):
            for j in range(rows):
                try:
                    if (img2[i, j][0] +
                        img2[i, j][1] +
                        img2[i, j][2]) <= 20:
                        roi[i, j] = roi[i, j]
                    else:
                        roi[i, j] = img2[i, j]
                except IndexError:
                    roi[i, j] = roi[i, j]
    else:
        img[dy:dy + rows, dx:dx + cols] = img2


# 加载指定目录下所有的人脸照片
def get_lableandwav(path, dir):
    global saveFace
    global known_face_encodings
    global known_face_names
    global ImagePath
    # 先暂停人脸识别
    saveFace = True
    allpath = []
    lllables = []
    dirs = os.listdir(path)
    for a in dirs:
        if os.path.isfile(path + "/" + a):
            allpath.append(dirs)
            if dir != "":
                lllables.append(dir)
                # 这里防止把本身图片存储根目录下的图片一并存起来
                if (path is not ImagePath):
                    image = face_recognition.load_image_file(path + '/' + a)
                    fe = face_recognition.face_encodings(image)
                    if(fe is not None) and (len(fe)>0):
                        known_face_encodings.append(fe[0])
                        known_face_names.append(path.split('/')[-1])

        else:
            get_lableandwav(str(path) + "/" + str(a), a)
        ##循环遍历这个文件夹
    saveFace = False
    return allpath, lllables


# 保存图片到本地
def savePic(img):
    name = str(time.time()) + '.jpg'
    b = cv2.imwrite(ImagePath+'/'+name, img)
    if (Debug):
        print('save %s,name=%s' % (str(b),name))


def show_camera(maskNet):
    index = 0
    global process_this_frame
    
    global known_face_encodings
    global nowFace

    face_names = []
    face_locations = []
    face_encodings = []
    face_mask_labs = []
    #下次出现人脸时是否保存的标记
    saveFace = False

    cap = cv2.VideoCapture(0)
    cap.set(3, CameraWidth)  # width=1280
    cap.set(4, CameraHeight)  # height=720
    task = time.time()
    while cap.isOpened():
        flag, img = cap.read()
        if img is None:
            break
        # 缩小图片，jetson跑不动全图
        small_frame = cv2.resize(img, (0, 0), fx=PictureZoom, fy=PictureZoom)
        # 判断有没有停止对人脸的识别
        if (saveFace):
            cv2.putText(img, "SaveFace....", (5, 18), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)
        else:
            if process_this_frame == 0:
                process_this_frame = 1
                
                face_locations = face_recognition.face_locations(small_frame, model='cnn')
                face_names = []
                face_mask_labs = []
                #如果没存过认识的人脸，那就统一不认识
                if(len(known_face_encodings)>0):
                    if face_locations is not None and len(face_locations) > 0 :
                        # 将人脸转换为特征点数组
                        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                        # 清空人名数组缓存
                    
                        for i in range(0, len(face_encodings)):
                            # 人脸特征点
                            face_encoding = face_encodings[i]
                            # 人脸的坐标点
                            (top, right, bottom, left) = face_locations[i]

                            isMask = False

                            #准备一个用来做口罩识别的人脸图像
                            mask_face = small_frame[top:bottom,left:right]
                            
                            #对人脸做处理，以便口罩识别哪里可以使用
                            mask_face = cv2.cvtColor(mask_face, cv2.COLOR_BGR2RGB)
                            mask_face = cv2.resize(mask_face, (224, 224))
                            mask_face = img_to_array(mask_face)
                            mask_face = preprocess_input(mask_face)
                            # 进行口罩预测
                            preds = maskNet.predict(np.array([mask_face], dtype="float32"), batch_size=32)
                            # 总共就一个人脸，你能测出来俩口罩？
                            mask = 0
                            withoutMask = 0
                            if len(preds)>0:
                                pred = preds[0]
                                #获取佩戴口罩的可能和没佩戴口罩的可能
                                (mask, withoutMask) = pred
                
                                isMask = mask > withoutMask
                            face_mask_labs.append(( (mask, withoutMask) ))

                            # 人脸识别的阈值，带口罩时自动降低些，不带时提高
                            confidence = 0.4
                            # 假如带口罩，提高阈值
                            if (isMask):
                                confidence = 0.6

                            # 先对比一下，这个脸和库里哪些脸匹配
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, confidence)
                            name = "Unknown"

                            ##如果在已知的_face_编码中发现匹配，只需使用第一个。
                            # 如果在匹配项中为True：
                            #     first_match_index = matches.index(True)
                            #     name = known_face_names[first_match_index]
                            # 或者，使用与新面距离最小的已知面
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)

                            # 从名字库中把最符合条件的名取出来
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                            face_names.append(name)
                else:
                    face_names.append("Unknown")
                    face_mask_labs.append((0,1))

            else:
                process_this_frame = 0

        # 获取原图的宽高
        rows, cols, channels = img.shape
        # 进行宽度标记，防止俩个人脸叠罗汉
        nowWidth = 0
        # 准备写字用的字体
        font = cv2.FONT_HERSHEY_DUPLEX

        #x轴的偏移量
        moveX = cols/3
        #y轴的偏移量
        moveY = rows/4
        #x轴的中心位置
        centX = cols/2
        #y轴的中心位置
        centY = rows/2
        for (top, right, bottom, left), name, (mask, withoutMask) in zip(face_locations, face_names,face_mask_labs):


            # 扩大一下边框，如果可以的话
            if (top >= 10):
                top -= 10
            if (rows >= bottom + 10):
                bottom = bottom + 10
            if (left >= 10):
                left -= 10
            if (cols >= right + 10):
                right += 10
            # 只把识别出的人的人脸绘制到左上角，未识别成功的不绘制
            if (name != "") and (name != "Unknown"):
               

                # 头像截取出来
                face = small_frame[top:bottom, left:right]

                # 提前准备一下原图
                m = (int)(min(rows, cols) / 5)

                # 重新设置一次头像大小，别有的大有的小,设置为正方形
                face = cv2.resize(face, (m, m))
                # 把人头画上去，先不给透明度，透明哪里有待优化
                drawableImgToImg(img, face, dx=nowWidth)
                # 把人名写头像下面,截取一下长度不然放不下
                cv2.putText(img, name[0:3], (nowWidth, m + 20), font, 1.0, (0, 0, 0), 1)
                # 增大下一个人脸绘制的起始位置，多给个1，给个间隔
                nowWidth += m + 1
            #没佩戴口罩的给予文字提示
            color = (0, 0, 255)

            

            isMask = mask > withoutMask
            if(isMask):
                color = (0,255,0)
                name = name + " mask" 
            else:
                name = name + " no mask" 
            name = "{}: {:.2f}%".format(name, max(mask, withoutMask) * 100)

            # 把坐标扩大，因为上面获取到的坐标是压缩后图片中的坐标
            zoom = (int)(1 / PictureZoom)
            left = (int)(left * zoom)
            top = (int)(top * zoom)
            right = (int)(right * zoom)
            bottom = (int)(bottom * zoom)

            #计算当前的位置
            nowX=(left+right)/2
            nowY=(top+bottom)/2
            #移动方向
        
            move = ""
            #判断移动方向
            if(nowX<moveX):
                #太靠左
                move=(" move to right")
            elif(nowX>(cols-moveX)):
                #太靠右
                move=(" move to left")
            else:
                move=""
            if(nowY<moveY):
                #太靠上
                move=move+(" move to bottom")
            elif(nowY>(rows-moveY)):
                #太靠下
                move=move+(" move to top")
            else:
                move=move+""
        
            print(left,right,top,bottom)

            if(move != ""):
                print(move)

            if(saveFace):
                # 手动存储图片
                saveFace = False
                face = img[top:bottom, left:right]
                savePic(face)
                cv2.putText(img, 'saveFace.....', (20,20), font, 1.0, (255,255,255), 2)

            # 绘制人脸的边框和人的名字
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)

            cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

            cv2.putText(img, name+move, (left + 6, bottom - 6), font, 1.0, (255,255,255), 2)
            
        fps = round(1/(abs(time.time()-task)),2)
        
        cv2.putText(img, str(fps), (40, rows- 100), font, 1.0, (0,0,0), 1)

        task = time.time()
            
        cv2.imshow("CSI Camera", img)

        kk = cv2.waitKey(1)
        if kk == ord('q'):  # 按下 q 键，退出
            break
        if kk == ord('s'):
            saveFace = True #准备存储下次出现时的人脸
        


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-z", "--pic_zoom", type=float, default=PictureZoom,
                    help="请输入图片处理过程中的压缩为原图的几倍，默认0.25")
    ap.add_argument("-m", "--mask_model", type=str, default="mask_detector.model",
                    help="口罩的模型名")
    ap.add_argument("-mc", "--mask_confidence", type=float, default=MaskConfidence,
                    help="识别度为多少是可以认定为佩戴口罩")
    ap.add_argument("-v", "--video_path", type=str, default="",
                    help="视频路径，如果没有，则默认启动0号位摄像头")

    ap.add_argument("-fip", "--face_image_path", type=str, default=ImagePath,
                    help="存储人的脸的路径")

    ap.add_argument("-ch", "--camera_height", type=int, default=CameraHeight,
                    help="摄像机的高")
    ap.add_argument("-cw", "--camera_width", type=int, default=CameraWidth,
                    help="摄像机的宽")
    args = vars(ap.parse_args())

    PictureZoom = args["pic_zoom"]
    MaskConfidence = args["mask_confidence"]
    videoPath = args["video_path"]
    ImagePath = args["face_image_path"]
    CameraHeight = args["camera_height"]
    CameraWidth = args["camera_width"]

    # 初始化tensorflow
    print('loading tensorflow...')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)
    
    maskNet=load_model(args['mask_model'])
    print('load tensorflow surcess')

    # 加载已知的人脸
    print('loading face.....')
    known_face_encodings = []
    known_face_names = []
    get_lableandwav(ImagePath, '')

    print('load surcess ')


    show_camera(maskNet)
