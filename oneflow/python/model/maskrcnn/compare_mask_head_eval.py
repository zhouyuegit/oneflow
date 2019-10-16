import numpy as np


def compare_numpy(x, y):
    assert x.shape == y.shape
    print("max absolute diff: {}".format(np.max(np.abs(x - y))))


print("compare level_idx_1")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/mask_head/mask_head_level_idx_0.(28,).npy"
)
y = np.load("eval_dump/mask_head_level_idx_2.npy")
compare_numpy(x, y)

print("compare level_idx_2")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/mask_head/mask_head_level_idx_1.(3,).npy"
)
y = np.load("eval_dump/mask_head_level_idx_3.npy")
compare_numpy(x, y)

# print("compare level_idx_3")
# x = np.load(
#     "/home/xfjiang/rcnn_eval_fake_data/iter_0/mask_head/mask_head_level_idx_2.(0,).npy"
# )
# y = np.load("eval_dump/mask_head_level_idx_4.npy")

print("compare level_idx_4")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/mask_head/mask_head_level_idx_3.(3,).npy"
)
y = np.load("eval_dump/mask_head_level_idx_5.npy")
compare_numpy(x, y)


print("compare roi_with_img_id_1")
x = np.load(
    "/home/xfjiang/rcnn_eval_fake_data/iter_0/mask_head/mask_head_rois_0.(28, 5).npy"
)
y = np.load("eval_dump/roi_with_img_id_0.npy")
compare_numpy(x, y)