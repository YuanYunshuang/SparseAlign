
train_hooks = [
        dict(type='MemoryUsageHook'),
        dict(type='TrainTimerHook'),
        dict(type="CheckPointsHook", epoch_every=10)
    ]


def get_test_nms_eval_hooks(point_cloud_range_test):
    return [
            dict(type='EvalInfoLoggingHook'),
            dict(type="DetectionNMSHook", nms_thr=0.01, pre_max_size=500, det_key='detection'),
            # dict(type="DetectionNMSHook", nms_thr=0.1, pre_max_size=500, det_key='detection_local'),
            dict(type="EvalDetectionBEVHook", save_result=False, global_sort=True,
                 pc_range=point_cloud_range_test),
        ]


def get_test_bev_semseg_hooks(point_cloud_range_test):
    return [dict(type='EvalBEVSemsegHook', test_range=point_cloud_range_test, save_result=True)]


def get_detection_plot(point_cloud_range_test, data_keys=['detection', 'global_labels']):
    return dict(title='DetectionCanvas', lidar_range=point_cloud_range_test,
                width=10, height=4, nrows=1, ncols=1, data_keys=data_keys)