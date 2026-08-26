from scripts.experiments.run_stage_a_gpu_scheduler import available_gpus


def test_available_gpus_respects_per_gpu_capacity():
    running = {
        "a1": {"gpu": 1},
        "a2": {"gpu": 1},
        "a3": {"gpu": 2},
    }

    assert available_gpus((1, 2, 3), running, jobs_per_gpu=2) == [2, 3]
