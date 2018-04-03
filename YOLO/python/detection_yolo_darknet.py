
# coding: utf-8

# In[252]:

from ctypes import *
import math
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("../libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int


predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]


network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, save_file_dir, file_name, colors, thresh=.3, hier_thresh=.3, nms=0.5):
    name = []
    prob = []
    bounding_box = []
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms):
        do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                # name.append(meta.names[i])
                name.append(i)
                prob.append(dets[j].prob[i])
                bounding_box.append((b.x, b.y, b.w, b.h))
                #res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                res.append((i, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    plt_bboxes(image, name, prob, bounding_box, save_file_dir, file_name, colors)
    return res, name, prob, bounding_box


def plt_bboxes(img_path, classes, scores, bboxes, save_file_dir, file, colors, figsize=(10, 10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    img = mpimg.imread(str(img_path)[2:-1])
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    for i in range(len(classes)):
        cls_id = classes[i]
        score = scores[i]
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        rect = plt.Rectangle((bboxes[i][0] - bboxes[i][2] / 2, bboxes[i][1] - bboxes[i][3] / 2), bboxes[i][2],
                             bboxes[i][3], fill=False,
                             linewidth=linewidth, edgecolor=colors[cls_id])
        plt.gca().add_patch(rect)
        plt.gca().text(bboxes[i][0] - bboxes[i][2] / 2, bboxes[i][1] - bboxes[i][3] / 2 - 2,
                       '{} | {:.3f}'.format(str(meta.names[cls_id], 'utf-8'), score),
                       fontsize=12, color='white')

    plt.axis('off')
    # plt.show()
    plt.savefig(save_file_dir + file[:-4] + '_detected.jpg', bbox_inches='tight')
    print('Saved:' + file[:-4] + '_detected.jpg')


net = load_net(b"../cfg/yolov3.cfg", b"../yolov3.weights", 0)
meta = load_meta(b"../cfg/coco.data")

original_images = '/Users/jingyi/study/video_test/video_image_set/test3_part6'
detected_images = '/Users/jingyi/study/video_test/video_image_set/test3_detected_part6/'
if not os.path.exists(detected_images):
    os.makedirs(detected_images)

colors = dict()
for root, dirs, files in os.walk(original_images, topdown=True):
    for file in files:
        if file.endswith(".jpg"):
            file_dir = os.path.join(root, file)
            res, name, prob, bounding_box = detect(net, meta, bytes(file_dir, encoding='utf-8'), detected_images, file, colors)
