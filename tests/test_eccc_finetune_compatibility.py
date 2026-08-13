from granitewxc.models import cordex_finetune_model
from granitewxc.models import eccc_finetune_model


def test_original_eccc_import_path_reexports_current_implementation() -> None:
    assert (
        eccc_finetune_model.ClimateDownscaleFinetuneModel
        is cordex_finetune_model.ClimateDownscaleFinetuneModel
    )
    assert (
        eccc_finetune_model.ClimateDownscaleFinetuneUNETModel
        is cordex_finetune_model.ClimateDownscaleFinetuneUNETModel
    )
    assert (
        eccc_finetune_model.ClimateECCCFinetuneWrapper
        is cordex_finetune_model.ClimateECCCFinetuneWrapper
    )
