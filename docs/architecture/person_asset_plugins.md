# Shared person assets, Qwen, and PACT

`person_preprocessing` owns the single canonical person-fit image geometry used
by SALT-VI, Qwen annotation, and PACT. It consumes the audited V1 + YOLO26 final
gate and emits a complete immutable 512x256 image inventory.

```text
audited V1 + YOLO26 gate
  YOLO26 detect/pose + YOLO11 cross-check + SCHP parsing
                         |
                         v
              person_preprocessing
              final image + geometry
                    |          |
                    v          v
          qwen_imagination    PACT
                    \          /
                     SALT training
```

The package cannot import SALT core, Qwen, or PACT. Passing records use their
audited bbox; fallback records retain the full source frame. The materializer
enforces exact key coverage, expected dataset counts, in-bounds boxes, and the
known partial-person failure invariant before writing any manifest.

SALT core reads the assets only when a training config sets
`prepared_data_root`. All three canonical Stage A person-fit configs point to
`/home/lab929/ybj/datasets/person-assets-512x256/person_fit`.
