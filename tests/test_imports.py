"""Garante que módulos do app importam sem depender de scripts/."""
from __future__ import annotations

import importlib


def test_import_modulos_raiz():
    for name in (
        "brasileirao_secoes",
        "brasileirao_sheet_names",
        "brasileirao_weekly_base",
        "brasileirao_projecao_core",
        "modelos_acumulados",
        "recency",
    ):
        importlib.import_module(name)


def test_sheet_names_reexport_labels():
    from brasileirao_secoes import LABEL_MEDIA as media_secao
    from brasileirao_sheet_names import LABEL_MEDIA, LABEL_PROB

    assert LABEL_MEDIA == media_secao
    assert LABEL_PROB == "Probabilístico"
