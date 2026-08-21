# Centralized Qwen imagination plugins

This directory is the sole source location for versioned Qwen imagination
implementations used by SALT-VI. SALT training code calls the stable bridge in
src/salt_vi/imagination.py and does not import QRI internals directly.

The qwen_imagination/api.py file defines the request and result contract.
The registry selects qri-v1, qri-v2, or a future version lazily.
The versions directory contains one adapter per version.
The regional directory contains shared regional engine code.
The configs and tests remain inside this plugin area.

Model weights, third-party source repositories, and experiment outputs are
runtime assets, not plugin source. They remain outside this directory.

The existing QRI checkout remains a migration baseline until both versions
pass the contract and regional tests.
