from deep_privacy.inference import deep_privacy_anonymizer, infer

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

if __name__ == "__main__":
    generator, imsize, source_path, image_paths, save_path, config = infer.read_args(
        additional_args=[{"name": "generator_batch_size", "default": 128},
                         {"name": "keypoint_batch_size", "default": 16},
                         {"name": "save_debug", "default": False}])

    if type(config.save_debug) != bool:
        save_debug = str2bool(config.save_debug)
    else:
        save_debug = config.save_debug

    print("save_debug:", save_debug, type(save_debug))

    anonymizer = deep_privacy_anonymizer.DeepPrivacyAnonymizer(
        generator, int(config.generator_batch_size), use_static_z=True,
        save_debug=save_debug, replace_tight_bbox=True,
        keypoint_batch_size=int(config.keypoint_batch_size))

    anonymizer.anonymize_folder(source_path, save_path)
