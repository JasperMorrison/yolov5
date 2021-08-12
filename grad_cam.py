"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path

from seaborn.matrix import heatmap

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from skimage import io

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box,xywh2xyxy
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync
from models.yolo import Detect
from models.common import Conv

class InputGradExtractor():
    def __init__(self, layer):
        self.layer = layer

    def register_hook(self):
        self.gradients = []
        self.handlers = []
        self.handlers.append(self.layer.register_hook(self._get_grads_hook))

    def _get_grads_hook(self, grad):
        self.gradients.append(grad[0])

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

class ConvCamExtractor():
    def __init__(self, layer):
        self.layer = layer

    def register_hook(self):
        self.features = []
        self.gradients = []
        self.handlers = []
        self.handlers.append(self.layer.register_forward_hook(self._get_features_hook))
        self.handlers.append(self.layer.register_full_backward_hook(self._get_grads_hook))

    def _get_features_hook(self, module, input, output):
        self.features.append(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradients.append(output_grad[0])

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

class DetectCamExtractor():
    def __init__(self, detect_layer):
        self.detect_layer = detect_layer
        self.nl = detect_layer.nl
        self.m = detect_layer.m

    def register_hook(self):
        self.features = []
        self.gradients = []
        self.handlers = []
        for i in range(self.nl):
            self.handlers.append(self.m[i].register_forward_hook(self._get_features_hook))
            self.handlers.append(self.m[i].register_full_backward_hook(self._get_grads_hook))

    def _get_features_hook(self, module, input, output):
        self.features.append(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradients.append(output_grad[0])

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # resize to image size
    h,w,c = image.shape
    mask = cv2.resize(mask, (w,h))
    # print("gen_cam, image and mask shape:", image.shape, mask.shape)
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    heatmap = heatmap[..., ::-1] # bgr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap).astype(np.uint8)

def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def grad_cam(gradients, features):
    cams = []
    for i,gradient in enumerate(reversed(gradients)):
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = features[i%3][0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        cams.append(cam)

    max_h, max_w = cams[0].shape

    cam_sum = np.zeros((max_h, max_w), np.float)

    for cam in cams:
        cam = cv2.resize(cam, (max_w, max_h))
        cam_sum += cam

    cam = cam_sum
    # 数值归一化
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def conv_grad_cam(gradients, features):
    cams = []
    normal = len(features)
    for i,gradient in enumerate(gradients):
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]

        feature = features[i%normal][0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        cam = np.maximum(cam, 0)  # ReLU

        cams.append(cam)

    max_h, max_w = cams[0].shape

    cam_sum = np.zeros((max_h, max_w), np.float)
    for cam in cams:
        cam = cv2.resize(cam, (max_w, max_h))
        cam_sum += cam

    cam = cam_sum
    # 数值归一化
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def grad_camplusplus(gradients, features):
    cams = []
    for i,gradient in enumerate(reversed(gradients)):
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = features[i%3][0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        cams.append(cam)

    max_h, max_w = cams[0].shape

    cam_sum = np.zeros((max_h, max_w), np.float)

    for cam in cams:
        cam = cv2.resize(cam, (max_w, max_h))
        cam_sum += cam

    cam = cam_sum
    # 数值归一化
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def conv_grad_camplusplus(gradients, features):
    cams = []
    normal = len(features)
    for i,gradient in enumerate(gradients):
        gradient = gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W]

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = features[i%normal][0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        cams.append(cam)

    max_h, max_w = cams[0].shape

    cam_sum = np.zeros((max_h, max_w), np.float)

    for cam in cams:
        cam = cv2.resize(cam, (max_w, max_h))
        cam_sum += cam

    cam = cam_sum
    # 数值归一化
    cam -= np.min(cam)
    cam /= np.max(cam)
    return cam

def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.cpu().data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def gen_input_grad(gradients):
    total_gradient = None
    for i,gradient in enumerate(gradients):
        if total_gradient is not None:
            total_gradient += gradient
        else:
            total_gradient = gradient
    return total_gradient

def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), None  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    detect_layer = None
    conv_layer = []
    # print model info
    for i,n_m in model.named_modules():
        n_m.requires_grad = True
        if isinstance(n_m, Conv): #Conv
            conv_layer.append(n_m)
        if isinstance(n_m, Detect):
            detect_layer = n_m
            anchors = n_m.anchors * n_m.stride.view(-1, 1, 1)
            anchors = anchors.cpu().numpy().astype(int)
            #print("anchors", n_m.anchors * n_m.stride.view(-1, 1, 1))
            print("anchors", anchors.flatten())

    set_requires_grad(model, True)
    torch.autograd.set_detect_anomaly(True)
    cam_extractor = DetectCamExtractor(detect_layer)
    conv_cam_extractor = []
    for index, conv in enumerate(conv_layer):
        conv_cam_extractor.append(ConvCamExtractor(conv))

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    import numpy as np
    for path, img, im0s, vid_cap in dataset:
        print()
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        cam_extractor.register_hook()
        for conv in conv_cam_extractor:
            conv.register_hook()
        t1 = time_sync()

        img.requires_grad = True
        # img.grad.zero_()
        model.zero_grad()
        input_extractor = InputGradExtractor(img)
        input_extractor.register_hook()

        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred_grad = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred_grad = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred_grad, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Process grad_cam
        for x_i in pred_grad:
            # Compute conf
            x = x_i.clone()
            # Do not mul.
            # If you only want to see foreground cam, only use x_i[:, 4:5] as conf.
            #x[:, 5:] = x_i[:, 5:] * x_i[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            # box = x[:, :4]
            # Detections matrix nx6 (xyxy, conf, cls)
            conf, j = x[:, 5:].max(1, keepdim=True)
            # Forground cam only
            # conf = x[:, 4:5]
            x_cat = torch.cat((box, conf, j), 1)[conf.view(-1) > conf_thres]

            print("NMS...")
            select_pred = []
            for p_det in x_cat:
                for det in pred[0]:
                    if not False in (det[0:4] == p_det[0:4]): # we do NMS
                        select_pred.append(p_det)
            print("NMS Done")
            # select_pred = x_cat
            for i, det in enumerate(select_pred):  # detections per image
                # Write results
                box = det[0:4] #xywh or xyxy
                conf = det[4]
                det_cls = det[5]
                if classes is not None and det_cls not in classes:
                    continue
                print("Backward with class", det_cls)
                # grad-cam
                # for item in box:
                #     item.backward(retain_graph=True)
                conf.backward(retain_graph=True)
                # det_cls.backward(retain_graph=True)

        # for grad in cam_extractor.gradients:
        #     print("=======grad shape======", grad.shape)

        # for feature in cam_extractor.features:
        #     print("=======feature shape======", feature.shape)

        cam_extractor.remove_handlers()
        for conv in conv_cam_extractor:
            conv.remove_handlers()
        input_extractor.remove_handlers()

        if len(cam_extractor.gradients) == 0:
            print("ERROR: cam_extractor no grad found!")
            continue

        print("Generate CAM...")

        import os
        cam_output = "./cam_output/"
        os.makedirs(cam_output, exist_ok=True)

        grad_cam_img = gen_cam(im0s, grad_cam(cam_extractor.gradients, cam_extractor.features))
        cam_path = cam_output + os.path.basename(path) + "_cam.jpg"
        heatmap_path = cam_output + os.path.basename(path) + "_heatmap.jpg"
        io.imsave(cam_path, grad_cam_img[0])
        io.imsave(heatmap_path, grad_cam_img[1])

        grad_camplusplus_img = gen_cam(im0s, grad_camplusplus(cam_extractor.gradients, cam_extractor.features))
        cam_path = cam_output + os.path.basename(path) + "_cam++.jpg"
        heatmap_path = cam_output + os.path.basename(path) + "_heatmap++.jpg"
        io.imsave(cam_path, grad_camplusplus_img[0])
        io.imsave(heatmap_path, grad_camplusplus_img[1])

        for index, conv in enumerate(conv_cam_extractor):
            conv_grad_cam_img = gen_cam(im0s, conv_grad_cam(conv.gradients, conv.features))
            cam_path = cam_output + os.path.basename(path) + "_" + str(index) + "_convcam.jpg"
            heatmap_path = cam_output + os.path.basename(path) + "_" + str(index) + "_convheatmap.jpg"
            io.imsave(cam_path, conv_grad_cam_img[0])
            io.imsave(heatmap_path, conv_grad_cam_img[1])

        for index, conv in enumerate(conv_cam_extractor):
            conv_grad_cam_img = gen_cam(im0s, conv_grad_camplusplus(conv.gradients, conv.features))
            cam_path = cam_output + os.path.basename(path) + "_" + str(index) + "_convcam++.jpg"
            heatmap_path = cam_output + os.path.basename(path) + "_" + str(index) + "_convheatmap++.jpg"
            io.imsave(cam_path, conv_grad_cam_img[0])
            io.imsave(heatmap_path, conv_grad_cam_img[1])

        input_grad = img.grad[0]  # [3,H,W]
        # input_grad = gen_input_grad(input_extractor.gradients)
        input_grad_img = norm_image(gen_gb(input_grad))
        cam_path = cam_output + os.path.basename(path) + "_gb.jpg"
        io.imsave(cam_path, input_grad_img)

       # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
