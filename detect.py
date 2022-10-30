import os, cv2

import line_fitting as lf
import cross_points as cp

from dataloader import *
import torch
import argparse
from pathlib import Path
import glob
from itertools import chain

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class detector(object):
    def __init__(self, weights, source, device, view_img, project):
        self.weights = weights
        self.source = source
        self.device = device
        print(device)
        self.view_img = view_img
        if os.path.exists(project)==False:
            os.mkdir(project)
        self.project = project
        self.result = dict.fromkeys(["lane_type", "lane_color", "lane_point"])

    def run(self):
        if os.path.isdir(self.source):
            print("start detect")
            img_list = glob.glob(os.path.join(self.source, "*.jpg"), recursive=True)
            self.model = torch.load(self.weights, map_location=self.device)
            self.model.eval()
            n = len(img_list)
            print(n)
            for i, img in enumerate(img_list[:5]):
                print(f"detect {i} / {n}")
                print(os.path.basename(img))
                mixed_im, res_img, lines= self.test_one_img(img)
                if mixed_im is False:
                    print("no line is found")
                    continue
                else:
                    cv2.imwrite(os.path.join(self.project, f"{os.path.basename(img).split('.')[0]}.jpg"),
                                cv2.cvtColor(mixed_im, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(self.project, f"fitted_{os.path.basename(img).split('.')[0]}.jpg"),
                                cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))

                    # 需要保存这些信息的话，先创建lane_points和new_result文件夹

                    # with open(os.path.join("lane_points", f"{os.path.basename(img).split('.')[0]}.txt"),"w+") as f:
                    #     lines = list(chain.from_iterable(lines))
                    #     f.write(" ".join(str(point) for point in lines))
                    #
                    # with open(os.path.join("new_result", f"{os.path.basename(img).split('.')[0]}.txt"),"w+") as f:
                    #     f.write(self.result["lane_type"]+"\n")
                    #     f.write(self.result["lane_color"] + "\n")
                    #     f.write(self.result["lane_point"] + "\n")


    def gen_mask(self, rgb_img, seg_label):
        # seg_label = cv2.cvtColor(seg_label, cv2.COLOR_RGB2BGR)
        alpha = 0.6
        mixed_im = cv2.addWeighted(seg_label, alpha, rgb_img, 1-alpha, 0)
        return mixed_im

    def gen_single_instance(self, binary_seg_pred, instance_seg_logits):
        w, h = binary_seg_pred.shape[2], binary_seg_pred.shape[1]
        # # 多个instance中挑出概率最大的
        # print(np.unique(instance_seg_logits).shape)
        # instance_seg_logits = np.argmax(instance_seg_logits, axis=0)  # (256, 512)
        # print(np.unique(instance_seg_logits))
        seg_imgs = []
        for i in range(instance_seg_logits.shape[0]):
            temp_seg_img = np.where(instance_seg_logits[i] != 0, binary_seg_pred[0],
                                    0)  # TODO:如何融合instance_seg_logits 和 binary_seg_pred
            # temp_seg_img = np.where(instance_seg_logits[i] != 0, 1, 0)  # TODO:如何展示单个instance
            temp_seg_img = temp_seg_img.reshape((h, w, 1))
            seg_img = np.concatenate((temp_seg_img, temp_seg_img, temp_seg_img), axis=2).astype(np.uint8) * 255
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_RGB2GRAY)
            seg_imgs.append(seg_img)
            # # 展示单个实例
            # plt.ion()
            # plt.figure()
            # plt.imshow(seg_img)
            # plt.show()
            # plt.pause(0.5)
            # plt.close()
        return seg_imgs


    def gen_binary_seg(self, binary_seg_pred):
        w, h = binary_seg_pred.shape[2], binary_seg_pred.shape[1]
        binary_seg_pred = binary_seg_pred.reshape((h, w, 1))
        binary_seg_pred = binary_seg_pred * 255
        binary_seg_pred = np.concatenate((binary_seg_pred, binary_seg_pred, binary_seg_pred), axis=2).astype(np.uint8)
        binary_seg_pred = cv2.cvtColor(binary_seg_pred, cv2.COLOR_RGB2GRAY)
        # gaus = cv2.GaussianBlur(binary_seg_pred, (3, 3), 0)
        # edges = cv2.Canny(gaus, 50, 150, apertureSize=3)
        # binary_seg_pred = cv2.resize(binary_seg_pred, (1920, 1080), cv2.INTER_NEAREST)
        # # 展示二值图
        # plt.ion()
        # plt.figure()
        # plt.imshow(binary_seg_pred)
        # plt.show()
        # plt.pause(0.5)
        # plt.close()
        return binary_seg_pred

    def gen_color_class(self, seg_logits):
        yellow_seg = np.where(np.argmax(seg_logits, axis=0) == 1, 1, 0)
        yellow_seg = yellow_seg[:, :, np.newaxis]
        yellow_seg = np.concatenate((yellow_seg, yellow_seg, yellow_seg), axis=2).astype(np.uint8)
        yellow_seg = cv2.cvtColor(yellow_seg, cv2.COLOR_RGB2GRAY)

        white_seg = np.where(np.argmax(seg_logits, axis=0) == 2, 1, 0)
        white_seg = white_seg[:, :, np.newaxis]
        white_seg = np.concatenate((white_seg, white_seg, white_seg), axis=2).astype(np.uint8)
        white_seg = cv2.cvtColor(white_seg, cv2.COLOR_RGB2GRAY)

        lane_color = [0, 0]  # 左右为白线
        if white_seg.shape[0] == 0:
            lane_color = [1, 1]
        elif yellow_seg.shape[0] == 0:
            lane_color = [0, 0]
        else:
            if lf.get_lane_slope(white_seg)[0] == 1:
                lane_color[0] = 0
            else:
                lane_color[1] = 0
            if lf.get_lane_slope(yellow_seg)[1] == 0:
                lane_color[1] = 1
            else:
                lane_color[0] = 1
        return lane_color

    def gen_seg(self, seg_logits):
        seg = np.where(np.argmax(seg_logits, axis=0) != 0, 1, 0)
        seg = seg[:, :, np.newaxis]
        seg = np.concatenate((seg, seg, seg), axis=2).astype(np.uint8)
        seg = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)
        return seg

    def test_one_img(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1920, 1080))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        trans_res = valid_transform(image=img)
        test_img = trans_res['image']
        test_img = test_img.unsqueeze(0)
        # print(test_img.shape)  # torch.Size([3, 256, 512])
        with torch.set_grad_enabled(False):
            # 预测
            test_img = test_img.to(self.device)
            net_output = self.model(test_img)  # dict_keys(['instance_seg_logits', 'binary_seg_pred', 'seg_logits'])
            # 对检测结果进行可视化
            seg_logits = net_output["seg_logits"].cpu().numpy()[0]
            instance_seg_logits = net_output["instance_seg_logits"].cpu().numpy()[0]
            binary_seg_pred = net_output["binary_seg_pred"].cpu().numpy()[0]
            # print(instance_seg_logits.shape, binary_seg_pred.shape)  # (5, 256, 512) (1, 256, 512)
            # print(np.unique(binary_seg_pred))  # [0, 1]

            # 生成单个实例
            instance_segs = self.gen_single_instance(binary_seg_pred, instance_seg_logits)  #为列表
            binary_seg = self.gen_binary_seg(binary_seg_pred)

            # 更准确的线条：应该用seg_logits+edges(binary_seg得到的)
            seg = self.gen_seg(seg_logits)

            # 直线拟合
            seg = cv2.resize(seg, (1920, 1080), cv2.INTER_NEAREST)
            res_img, lines, solid_or_dotted = lf.lane_fitting(img, [seg])  # 返回的是标上线的图，以及线上两端点
            centre_x = 1920 // 2
            # print(lines)  #(((x1,y1),(x2,y2)),((x3,y3),(x4,y4)))
            # 将一边为空的车道线置为沿中心x轴翻转的对称道路线
            if lines[0] is None and lines[1] is not None:
                lines[0] = [[0, 0], [0, 0]]
                for i in range(2):
                    lines[0][i][0] = 2*centre_x - lines[1][i][0]
                    lines[0][i][1] = lines[1][i][1]
            elif lines[1] is None and lines[0] is not None:
                lines[1] = [[0, 0], [0, 0]]
                for i in range(2):
                    lines[1][i][0] = 2*centre_x - lines[0][i][0]
                    lines[1][i][1] = lines[0][i][1]
            elif lines[1] is None and lines[0] is None:
                return False, False, False

            self.result["lane_type"] = str(solid_or_dotted[0]) + str(solid_or_dotted[1])
            # lines[1][0][0] = 1920
            # lines[1][0][1] = 1080
            # 拟合成直线后取交点，810，1080，左上是原点
            # 用lines框出地标检测的ROI区域，输出图片有遮挡的test图像
            line1 = list(chain.from_iterable(lines[0]))
            line2 = list(chain.from_iterable(lines[1]))
            line_v1 = [0, 810, 1920, 810]
            line_v2 = [0, 1080, 1920, 1080]
            left_up = cp.get_line_cross_point(line1, line_v1)
            right_up = cp.get_line_cross_point(line2, line_v1)
            left_down = cp.get_line_cross_point(line1, line_v2)
            right_down = cp.get_line_cross_point(line2, line_v2)
            # print(left_up, right_up, left_down, right_down)
            lane_points = [left_up, right_up, left_down, right_down]
            self.result["lane_point"] = "\n".join([" ".join([str(item) for item in point]) for point in lane_points])


            # 背景为（0~50）黄色线为（51~200），白色线为（201~255）
            # 区分左右的黄白线
            # 生成黄白线实例
            lane_color = self.gen_color_class(seg_logits)
            # print(lane_color)
            self.result["lane_color"] = str(lane_color[0])+str(lane_color[1])

            # 可视化黄白线
            result = (np.argmax(seg_logits, axis=0) * 127).astype(np.uint8)  # 此处背景是0，黄色线是127，白色线是254
            result = result[:, :, np.newaxis]
            result = np.concatenate((result, result, result), axis=2).astype(np.uint8)  # (256, 512, 3)
            result = cv2.resize(result, (1920, 1080))
            mixed_im = self.gen_mask(img, result)
            if self.view_img:
                plt.ion()
                plt.subplot(311)
                plt.imshow(img)
                plt.subplot(312)
                plt.imshow(result)
                plt.subplot(313)
                plt.imshow(mixed_im)
                plt.show()
                plt.ion()
                plt.pause(1)  # 显示秒数
                plt.close()
        return mixed_im, res_img, lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'checkpoints/1000.pth', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'img_total', help='the image dir')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--project', default=ROOT / 'car_detect', help='save results to project/name')
    args = parser.parse_args()
    task = detector(**vars(args))
    task.run()
