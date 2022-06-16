import neptune.new as neptune

def neptune_initialize(args, config):
    neptune_callback = None
    if args.enable_log:
        neptune_callback = neptune.init(
            name = args.experiment_name,
            project = config['neptune_project'],
            api_token = config['neptune_api'],
            source_files = config['neptune_source_file']
        )
        neptune_callback["parameters"] = config
    
    return neptune_callback