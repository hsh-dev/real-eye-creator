CONFIG = {
    'save_path' : './blink_output',
    'batch_size': 64,
    'val_batch_size' : 64,
    'test_batch_size' : 64,
    'input_size' : (60, 36),
    'epochs' : 500,
    'cycle' : 4,
    'decay_steps' : 200,
    'learning_rate' : 1e-3,
    'min_learning_rate' : 1e-4,
    'loss_function' : "BFCE",
    'label_smoothing' : 0.1,
    
    '300vw_blink_path' : '/mnt/ssd1/blink_dataset/300vw_blink',
    'unity_eyes_path' : '/mnt/ssd1/blink_dataset/unity_dataset_new',
    'rt_bene_path' : '/mnt/ssd1/RT_BENE',
    'golflab_path' : '/mnt/ssd1/CREATZ_DATASET',

    'test_rt_bene_path' : '/mnt/ssd1/blink_testset/rt_bene_test',
    'test_golflab_path' : '/mnt/ssd1/CREATZ_DATASET',
    
    'rt_bene_train_subject' : [1, 2, 8, 10, 3, 4, 7, 9],
    'rt_bene_valid_subject' : [5, 12, 13, 14],
    'rt_bene_test_subject' : [0, 11, 15, 16],

    'golflab_train_subject' : [2, 5, 6, 9, 10, 12],
    'golflab_valid_subject' : [1, 4, 8],
    'golflab_test_subject' : [3, 11, 13],
    
    'train_dataset_list' : ["rt_bene", "unity_eyes", "golflab"],
    'valid_dataset_list' : ["rt_bene", "unity_eyes", "golflab"],
    'test_dataset_list' : ["rt_bene", "unity_eyes", "golflab"],

    'rt_bene_resize_ratio' : 1,
    'train_rt_bene_resize' : False, 
    'test_rt_bene_resize' : False,
    'train_unity_eyes_resize' : True,
    'test_unity_eyes_resize' : True,
    
    'neptune_project' : "vcamp/jeff-blink",
    'neptune_api' : "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YzkwOTk5Zi03ZGE1LTQ2MjQtYTkxNC0wZTNiN2I1Y2M5OTkifQ==",
    'neptune_source_file' : ["run.py","finetuning.py","config.py","branch_run.py"
                            "trainer/*.py",
                            "dataset/*.py",
                            "models/EyeStateClassifierModel.py"]    
    
}