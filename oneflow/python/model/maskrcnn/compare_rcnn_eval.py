import numpy as np


def compare_numpy(x, y):
    assert x.shape == y.shape
    print("max absolute diff: {}".format(np.max(np.abs(x - y))))


print("compare roi_feature_0")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/roi_head/roi_head_roi_feature_0.(662, 256, 7, 7).npy"
)
y = np.load("eval_dump/roi_feature_0.npy")
compare_numpy(x, y)

print("compare roi_feature_1")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/roi_head/roi_head_roi_feature_1.(135, 256, 7, 7).npy"
)
y = np.load("eval_dump/roi_feature_1.npy")
compare_numpy(x, y)

print("compare roi_feature_2")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/roi_head/roi_head_roi_feature_2.(134, 256, 7, 7).npy"
)
y = np.load("eval_dump/roi_feature_2.npy")
compare_numpy(x, y)


print("compare roi_feature_3")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/roi_head/roi_head_roi_feature_3.(69, 256, 7, 7).npy"
)
y = np.load("eval_dump/roi_feature_3.npy")
compare_numpy(x, y)

print("compare pooler result")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/roi_head/roi_head_pooler_result.(1000, 256, 7, 7).npy"
)
y = np.load("eval_dump/roi_features_reorder.npy")
compare_numpy(x, y)

print("compare bbox regression")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/roi_head/box_regression.(1000, 324).npy"
)
y = np.load("eval_dump/bbox_regression.npy")
compare_numpy(x, y)

print("compare cls logits")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/roi_head/class_logits.(1000, 81).npy"
)
y = np.load("eval_dump/cls_logits.npy")
compare_numpy(x, y)
