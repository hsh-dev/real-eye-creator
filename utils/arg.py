import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import argparse
from utils.dir import check_make_dir

def arg_init(CONFIG):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", 
        help="write path to save output")
    parser.add_argument("-n", "--experiment_name", 
        help="experiment name to save",
        default="new_test")
    parser.add_argument("-r", "--resize_ratio", help="ratio for rt_bene undersampling")
    parser.add_argument("--enable_train_resize", help="Enable rt_bene train resize", action='store_true')
    parser.add_argument("--enable_test_resize", help="Enable rt_bene test resize", action='store_true')
    parser.add_argument(
        "--enable_log",
        help="Decide whether to upload log on neptune or not",
        action='store_true'
    )
    parser.add_argument("-sub", "--subject", 
        nargs="+",
        help="subject list for golflab test"
    )
    args = parser.parse_args()
    
    output_path = os.path.abspath(CONFIG["save_path"])
    if args.save_path is not None:
        output_path = os.path.abspath(args.save_path)
    check_make_dir(output_path)

    CONFIG["save_path"] = os.path.join(output_path, args.experiment_name)
    check_make_dir(CONFIG["save_path"])

    if args.resize_ratio is not None:
        CONFIG["rt_bene_resize_ratio"] = int(args.resize_ratio)

    if args.enable_train_resize:
        CONFIG["train_rt_bene_resize"] = True
    
    if args.enable_test_resize:
        CONFIG["test_rt_bene_resize"] = True

    if args.subject is not None:
        CONFIG["golflab_test_subject"] = args.subject
    return args, CONFIG