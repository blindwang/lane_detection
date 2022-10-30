import cv2
import os, glob

img_root = 'car_detect'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 20    #保存视频的FPS，可以适当调整
size=(640, 480)
#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('car_deploy_video.mp4',fourcc,fps,size)#最后一个是保存图片的尺寸

# path_lane_ls = glob.glob(os.path.join(img_root, "*.jpg"))
path_yolo_ls = glob.glob(os.path.join("yolo_result", "*.jpg"))
# for path in path_lane_ls:
#     frame = cv2.imread(path)
#     videoWriter.write(frame)
# print(path_lane_ls)
print(path_yolo_ls)
for path in path_yolo_ls:
    frame = cv2.imread(path)
    videoWriter.write(frame)
videoWriter.release()