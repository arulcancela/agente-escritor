"""Tests de las funciones puras del motor (escritor_autonomo.py).

Se testean utilidades deterministas que no requieren red ni LLM:
detección de idioma, limpieza de Markdown, carga de configuración y
métricas del evaluador de calidad.
"""

from __future__ import annotations

import escritor_autonomo as motor


# ── detectar_idioma ──────────────────────────────────────────────────────────

def test_detecta_espanol() -> None:
    texto = "este es un documento de prueba que habla sobre el sistema y la gestion de datos para la empresa"
    assert motor.detectar_idioma(texto) == "es"


def test_detecta_ingles() -> None:
    texto = "this is a test document that talks about the system and the management of data for the company"
    assert motor.detectar_idioma(texto) == "en"


def test_idioma_por_defecto_es() -> None:
    # Texto ambiguo → por defecto español.
    assert motor.detectar_idioma("") == "es"


# ── limpiar_markdown ─────────────────────────────────────────────────────────

def test_limpiar_markdown_quita_bloque_markdown() -> None:
    entrada = "```markdown\n# Título\nContenido\n```"
    salida = motor.limpiar_markdown(entrada)
    assert salida == "# Título\nContenido"


def test_limpiar_markdown_quita_bloque_generico() -> None:
    entrada = "```\ncódigo o texto\n```"
    salida = motor.limpiar_markdown(entrada)
    assert salida == "código o texto"


def test_limpiar_markdown_sin_bloque_no_cambia() -> None:
    entrada = "# Título\n\nTexto normal"
    assert motor.limpiar_markdown(entrada) == entrada


# ── cargar_config ────────────────────────────────────────────────────────────

def test_cargar_config_devuelve_dict() -> None:
    cfg = motor.cargar_config()
    assert isinstance(cfg, dict)
    assert "modelo_agente" in cfg


def test_config_tokens_vacios_por_defecto() -> None:
    """La config pública NO debe contener tokens reales."""
    cfg = motor.cargar_config()
    assert cfg.get("github_token", "") == ""
    assert cfg.get("notion_token", "") == ""


# ── EvaluadorCalidad ─────────────────────────────────────────────────────────

def test_cobertura_sin_fuentes_es_cero() -> None:
    assert motor.EvaluadorCalidad.cobertura_fuentes("documento", []) == 0.0


def test_densidad_informativa_cuenta_datos() -> None:
    doc = "El proyecto costó 50% más en 2026-01-01 y empleó a Juan Pérez."
    densidad = motor.EvaluadorCalidad.densidad_informativa(doc)
    assert densidad > 0


def test_legibilidad_en_rango() -> None:
    score = motor.EvaluadorCalidad.legibilidad("Este es un texto sencillo. Tiene dos frases.")
    assert 0 <= score <= 100


def test_legibilidad_acepta_idioma() -> None:
    score = motor.EvaluadorCalidad.legibilidad("Short text here. Two sentences.", idioma="en")
    assert 0 <= score <= 100


# ── Estructura de fases del pipeline ─────────────────────────────────────────

def test_fases_del_pipeline_definidas() -> None:
    for fase in ("ingest", "research", "write", "review", "refine", "finalize"):
        assert fase in motor.PHASE_PROMPTS


def test_mermaid_templates_presentes() -> None:
    assert "flowchart" in motor.MERMAID_TEMPLATES
