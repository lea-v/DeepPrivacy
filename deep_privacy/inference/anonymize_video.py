from deep_privacy.inference import deep_privacy_anonymizer, infer

if __name__ == "__main__":
    generator, _, source_path, _, target_path, config = infer.read_args(
        [{"name": "anonymize_source", "default": False},
         {"name": "max_face_size", "default": 1.0},
         {"name": "without_source", "default": False},
         {"name": "generator_batch_size", "default": 32}
        ],
    )
    a = deep_privacy_anonymizer.DeepPrivacyAnonymizer(generator,
                                                      int(config.generator_batch_size),
                                                      use_static_z=True,
                                                      keypoint_threshold=.1,
                                                      face_threshold=.6,
                                                      replace_tight_bbox=True)

    a.anonymize_video(source_path,
                      target_path,
                      start_frame=0,
                      end_frame=None,
                      with_keypoints=True,
                      anonymize_source=config.anonymize_source,
                      max_face_size=float(config.max_face_size),
                      without_source=config.without_source)
