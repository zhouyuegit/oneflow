./debug_rcnn_eval.sh
cd /home/xfjiang/rcnn_eval_fake_data && ./fetch_data.sh && cd -
python compare_rcnn_eval.py
