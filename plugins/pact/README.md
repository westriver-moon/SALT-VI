# PACT plugin

PACT is a repository-owned plugin, not part of the SALT-VI core package. It
consumes immutable final images and raw COCO-17 pose records from
`person_preprocessing`; it never runs YOLO and never calls Qwen.

Its offline command derives PACT-specific head, torso, arms, and legs masks,
including masks projected to the configured ViT patch grid. Training-time
counterfactual intervention and losses belong here as the research method is
implemented. Inference remains the unchanged SALT-VI backbone.
