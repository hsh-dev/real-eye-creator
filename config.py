CONFIG = {
    'save_path' : './output',
    'batch_size': 1,
    'val_batch_size' : 8,
    'test_batch_size' : 1,
    'input_size' : (60, 36),
    'epochs' : 400,
    'cycle' : 20,
    'decay_steps' : 200,
    'learning_rate' : 3e-4,
    'min_learning_rate' : 1e-4,

    '300vw_blink_path' : '/mnt/ssd1/blink_dataset/300vw_blink',
    'rt_bene_path' : '/mnt/ssd1/blink_dataset/rt_bene_new',
    'unity_eyes_path' : '/mnt/ssd1/blink_dataset/unity_dataset_new',

    'test_rt_bene_path' : '/mnt/ssd1/blink_testset/rt_bene_test',
    'test_golflab_path' : '/mnt/ssd1/blink_testset/golflab_test'
}