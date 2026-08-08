from types import SimpleNamespace

from salt_vi.entrypoints.output_paths import build_experiment_name, resolve_run_directory


def config(**overrides):
    values = dict(
        DEBUG=False,
        DEBUG_DIR="/tmp/debug",
        auto_resume_training_from_lastest_step=False,
        resume_train_epoch=-1,
        mode="train",
        output_root="/runs/pasd",
        output_path=None,
        dataset="sysu",
        trial=1,
        training_mode="RGB_IR_Text",
        joint_mode="uni",
        captioner_name="PASD",
        fusion_way="parameter_add",
        llm_aug=False,
        llm_aug_prob=0.5,
        loss_names="id,cross_modal_hard",
        Return_B4_BN=False,
        uni_BN=False,
        Fix_Visual=True,
        Feat_Filter=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fresh_training_resolves_output_root_once():
    value = config()
    name = build_experiment_name(value)
    assert name == (
        "Baseline_train[RGB_IR_Text]_joint[uni]_PASD_parameter_add_"
        "id,cross_modal_hard_Fix_Visual"
    )
    assert resolve_run_directory(value) == f"/runs/pasd/sysu/FV/{name}"


def test_resume_and_test_use_final_output_path():
    final = "/runs/pasd/sysu/FV/experiment"
    assert resolve_run_directory(
        config(output_root=None, output_path=final, resume_train_epoch=3)
    ) == final
    assert resolve_run_directory(config(output_root=None, output_path=final, mode="test")) == final
