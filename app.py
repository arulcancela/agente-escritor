#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AGENTE ESCRITOR — Interfaz Web Streamlit                                ║
║  Pipeline de fases · Progreso en tiempo real · Subida de archivos           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Ejecutar:  streamlit run app.py
"""

import streamlit as st
import sys, os, json, time, shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escritor_autonomo import (AgenteEscritor, cargar_config, CONFIG,
                                   ROLES, MERMAID_TEMPLATES, ExportadorPDF,
                                   EvaluadorCalidad, detectar_idioma,
                                   PHASE_TO_ROLE)

# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Agente Escritor", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    /* ─── Tema general ────────────────────────────────────────── */
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header { font-size: 0.9rem; color: #888; margin-bottom: 1.5rem; }

    /* ─── Badges de rol ───────────────────────────────────────── */
    .role-badge {
        display: inline-block; padding: 3px 12px; border-radius: 12px;
        font-weight: 600; font-size: 0.78rem; margin: 2px 0;
    }
    .role-orquestador { background: #e8f5e9; color: #2e7d32; }
    .role-investigador { background: #e3f2fd; color: #1565c0; }
    .role-redactor { background: #fff3e0; color: #e65100; }
    .role-editor { background: #fce4ec; color: #c62828; }
    .role-traductor { background: #f3e5f5; color: #7b1fa2; }

    /* ─── Tarjetas de métricas ────────────────────────────────── */
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 14px; border-radius: 10px;
        text-align: center; box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    .metric-value { font-size: 1.9rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; opacity: 0.9; }

    /* ─── Barra de fase ───────────────────────────────────────── */
    .phase-bar {
        display: flex; gap: 4px; margin: 10px 0;
    }
    .phase-step {
        flex: 1; text-align: center; padding: 6px 4px;
        border-radius: 6px; font-size: 0.7rem; font-weight: 600;
    }
    .phase-done { background: #c8e6c9; color: #2e7d32; }
    .phase-active { background: #667eea; color: white; animation: pulse 1.5s infinite; }
    .phase-pending { background: #f5f5f5; color: #bbb; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* ─── Entrada de scratchpad ────────────────────────────────── */
    .scratch-entry {
        padding: 6px 10px; margin: 3px 0; border-radius: 6px;
        background: #f8f9fa; font-size: 0.8rem; border-left: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  ESTADO DE SESIÓN
# ═══════════════════════════════════════════════════════════════════════════
for key, default in [
    ('messages', []), ('docs_generados', []), ('log', []),
    ('idioma_ui', 'es'), ('proyecto_actual', None), ('agente', None),
    ('diagrama_actual', ''), ('tipo_diagrama', 'flowchart'),
    ('ultimo_resultado', ''), ('scratchpad', []),
    ('current_phase', 'ingest'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Agente Escritor")
    idioma = st.selectbox("🌐", ["🇪🇸 Español", "🇬🇧 English"],
                          index=0 if st.session_state.idioma_ui == 'es' else 1)
    st.session_state.idioma_ui = 'es' if 'Español' in idioma else 'en'
    t = lambda es, en: es if st.session_state.idioma_ui == 'es' else en
    st.markdown("---")

    # ─── Proyecto ─────────────────────────────────────────────
    proyectos_dir = Path(__file__).resolve().parent / "proyectos"
    proyectos_dir.mkdir(exist_ok=True)
    proyectos = sorted([d.name for d in proyectos_dir.iterdir()
                        if d.is_dir() and not d.name.startswith('.')])

    st.markdown(f"### 📁 {t('Proyecto', 'Project')}")
    opciones = ["+ " + t("Nuevo proyecto", "New project")] + proyectos
    proy = st.selectbox(t("Seleccionar", "Select"), opciones, label_visibility="collapsed")

    if proy.startswith("+ "):
        nombre_nuevo = st.text_input(t("Nombre", "Name"), placeholder="mi_proyecto")
        if nombre_nuevo and st.button(t("Crear", "Create")):
            (proyectos_dir / nombre_nuevo).mkdir(exist_ok=True)
            st.session_state.proyecto_actual = str(proyectos_dir / nombre_nuevo)
            st.session_state.agente = None
            st.rerun()
    else:
        nuevo_path = str(proyectos_dir / proy)
        if st.session_state.proyecto_actual != nuevo_path:
            st.session_state.proyecto_actual = nuevo_path
            st.session_state.agente = None
            st.session_state.docs_generados = []
            st.session_state.messages = []
            st.session_state.scratchpad = []
            st.session_state.log = []

    proyecto_path = st.session_state.proyecto_actual
    st.markdown("---")

    # ─── Subir fuentes ────────────────────────────────────────
    if proyecto_path:
        st.markdown(f"### 📎 {t('Subir fuentes', 'Upload sources')}")
        uploaded = st.file_uploader(
            t("Arrastra archivos aquí", "Drag files here"),
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'md', 'csv', 'json', 'html', 'xml', 'yaml', 'yml', 'log'],
            label_visibility="collapsed"
        )
        if uploaded:
            for f in uploaded:
                dest = Path(proyecto_path) / f.name
                dest.write_bytes(f.getvalue())
                st.success(f"✅ {f.name}")
            st.session_state.agente = None  # Reiniciar agente para re-indexar
        st.markdown("---")

    # ─── Fuentes existentes ───────────────────────────────────
    if proyecto_path:
        fuentes_dir = Path(proyecto_path)
        archivos = []
        for f in sorted(fuentes_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in ['.txt', '.md', '.pdf', '.docx', '.csv',
                                                       '.json', '.html', '.log', '.yaml', '.xml']:
                if 'resultado' not in str(f) and 'cache' not in str(f):
                    archivos.append(f)

        st.markdown(f"### 📂 {t('Fuentes', 'Sources')} ({len(archivos)})")
        if archivos:
            for a in archivos[:20]:
                rel = str(a.relative_to(fuentes_dir))
                st.text(f"📄 {rel[:50]}")
            if len(archivos) > 20:
                st.caption(f"... +{len(archivos)-20} {t('más', 'more')}")
        else:
            st.info(t("Sube archivos o añádelos a la carpeta del proyecto",
                      "Upload files or add them to the project folder"))
    st.markdown("---")

    # ─── Limpiar caché ────────────────────────────────────────
    if st.button(f"🧹 {t('Limpiar caché', 'Clear cache')}"):
        d = Path(proyecto_path) / "cache" if proyecto_path else None
        if d and d.exists():
            shutil.rmtree(d)
            st.session_state.agente = None
            st.success(t("Caché eliminado", "Cache cleared"))
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIÓN: BARRA DE FASES
# ═══════════════════════════════════════════════════════════════════════════

def render_phase_bar(current_phase, completed_phases):
    """Renderiza la barra de progreso por fases."""
    all_phases = ["ingest", "research", "write", "review", "refine", "finalize"]
    labels = {"ingest": "📥 Ingestión", "research": "🔍 Investigación",
              "write": "✍️ Redacción", "review": "🔎 Revisión",
              "refine": "🔧 Refinamiento", "finalize": "🏁 Final"}

    html = '<div class="phase-bar">'
    for phase in all_phases:
        if phase in completed_phases:
            cls = "phase-done"
        elif phase == current_phase:
            cls = "phase-active"
        else:
            cls = "phase-pending"
        html += f'<div class="phase-step {cls}">{labels.get(phase, phase)}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  ÁREA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

col_main, col_side = st.columns([3, 1])

with col_main:
    st.markdown('<div class="main-header">🧠 Agente Escritor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pipeline Multi-Agente · RAG Semántico · DeepSeek</div>',
                unsafe_allow_html=True)

    # Barra de fases
    completed = set()
    if st.session_state.scratchpad:
        for entry in st.session_state.scratchpad:
            p = entry.get('phase', '')
            if p and p != st.session_state.current_phase:
                completed.add(p)
    render_phase_bar(st.session_state.current_phase, completed)

    tabs = st.tabs([t("💬 Chat", "💬 Chat"),
                     t("📄 Documentos", "📄 Documents"),
                     t("📊 Diagramas", "📊 Diagrams"),
                     t("📈 Métricas", "📈 Metrics"),
                     t("📋 Scratchpad", "📋 Scratchpad")])

    # ── TAB CHAT ──────────────────────────────────────────────
    with tabs[0]:
        for msg in st.session_state.messages:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(msg['content'])
            elif msg['role'] == 'assistant':
                with st.chat_message("assistant", avatar="🧠"):
                    st.markdown(msg['content'][:3000])

        consulta = st.chat_input(t("Escribe tu objetivo...", "Write your goal..."))

        if consulta:
            if not proyecto_path:
                st.error(t("Selecciona o crea un proyecto primero",
                           "Select or create a project first"))
            else:
                st.session_state.messages.append({"role": "user", "content": consulta})

                cfg = cargar_config()
                if st.session_state.agente is None:
                    st.session_state.agente = AgenteEscritor(proyecto_path, cfg)

                agente = st.session_state.agente
                agente.iteracion = 0
                agente.documentos_generados = []
                agente.scratchpad = []
                log_entries = []

                def ui_callback(iteracion, herramienta, args, resultado):
                    log_entries.append({
                        'iter': iteracion,
                        'phase': agente.current_phase,
                        'rol': PHASE_TO_ROLE.get(agente.current_phase, 'orquestador'),
                        'accion': herramienta,
                        'resumen': str(resultado)[:200]
                    })

                agente._callbacks = [ui_callback]

                with st.status(
                    t("🤖 Agente trabajando...", "🤖 Agent working..."),
                    expanded=True
                ) as status:
                    try:
                        resultado = agente.ejecutar(consulta)
                        status.update(
                            label=t("✅ Completado", "✅ Completed"),
                            state="complete"
                        )
                    except Exception as e:
                        resultado = f"❌ Error: {e}"
                        status.update(label=f"❌ Error", state="error")

                    # Mostrar log de acciones dentro del status
                    for entry in log_entries:
                        phase = entry.get('phase', '?')
                        accion = entry.get('accion', '?')
                        resumen = entry.get('resumen', '')[:120]
                        st.write(f"**[{phase.upper()}]** 🔧 `{accion}`: {resumen}")

                st.session_state.messages.append({"role": "assistant", "content": resultado})
                st.session_state.docs_generados = agente.documentos_generados
                st.session_state.log = log_entries
                st.session_state.scratchpad = agente.scratchpad
                st.session_state.current_phase = agente.current_phase
                st.session_state.ultimo_resultado = resultado
                st.rerun()

    # ── TAB DOCUMENTOS ────────────────────────────────────────
    with tabs[1]:
        st.markdown(f"### 📄 {t('Documentos generados', 'Generated documents')}")
        docs = list(st.session_state.docs_generados)
        if not docs and proyecto_path:
            res_dir = Path(proyecto_path) / "resultado"
            if res_dir.exists():
                for f in sorted(res_dir.glob("*.md")):
                    if '.auditoria' not in f.name and '.verificacion' not in f.name:
                        try:
                            docs.append({"nombre": f.name, "ruta": str(f),
                                         "contenido": f.read_text("utf-8")})
                        except Exception:
                            pass

        if docs:
            for i, doc in enumerate(docs):
                with st.expander(f"📄 {doc['nombre']} ({doc.get('idioma', 'es')})", expanded=(i == 0)):
                    contenido = doc.get('contenido', '')
                    st.markdown(contenido)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button(f"📥 PDF", key=f"pdf_{i}"):
                            ruta_md = Path(proyecto_path) / "resultado" / doc['nombre']
                            if ruta_md.exists():
                                ruta_pdf = str(ruta_md).replace('.md', '.pdf')
                                ok, msg = ExportadorPDF.md_a_pdf(ruta_md.read_text("utf-8"), ruta_pdf)
                                st.success(msg) if ok else st.error(msg)
                    with c2:
                        if st.button(f"📥 DOCX", key=f"docx_{i}"):
                            ruta_md = Path(proyecto_path) / "resultado" / doc['nombre']
                            if ruta_md.exists():
                                ruta_docx = str(ruta_md).replace('.md', '.docx')
                                ok, msg = ExportadorPDF.md_a_docx(ruta_md.read_text("utf-8"), ruta_docx)
                                st.success(msg) if ok else st.error(msg)
                    with c3:
                        st.download_button(
                            "📥 MD",
                            data=contenido,
                            file_name=doc['nombre'],
                            mime="text/markdown",
                            key=f"dl_{i}"
                        )
        else:
            st.info(t("Genera un documento en el chat", "Generate a document in chat"))

    # ── TAB DIAGRAMAS ─────────────────────────────────────────
    with tabs[2]:
        st.markdown(f"### 📊 {t('Diagramas Mermaid', 'Mermaid Diagrams')}")
        c1, c2 = st.columns([1, 2])
        with c1:
            tipos = list(MERMAID_TEMPLATES.keys())
            tipo_sel = st.selectbox(t("Tipo", "Type"), tipos)
            descripcion = st.text_area(t("Descripción", "Description"),
                                       placeholder=t("Datos para el diagrama...",
                                                     "Data for diagram..."),
                                       height=120)
            if st.button(f"🎨 {t('Generar', 'Generate')}", use_container_width=True):
                if proyecto_path and descripcion:
                    cfg = cargar_config()
                    agente_temp = AgenteEscritor(proyecto_path, cfg)
                    from escritor_autonomo import generar_mermaid
                    with st.spinner(t("Generando diagrama...", "Generating diagram...")):
                        codigo = generar_mermaid(agente_temp.client, cfg, tipo_sel,
                                                 descripcion, st.session_state.idioma_ui)
                    st.session_state['diagrama_actual'] = codigo
                    st.session_state['tipo_diagrama'] = tipo_sel
                    st.rerun()
        with c2:
            codigo = st.session_state.get('diagrama_actual', '')
            if codigo:
                st.code(codigo, language=None)
                html_m = f"""<html><head>
                <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
                <script>mermaid.initialize({{startOnLoad:true,theme:'default'}});</script>
                </head><body style="background:white;"><div class="mermaid">{codigo}</div></body></html>"""
                st.components.v1.html(html_m, height=400, scrolling=True)
            else:
                st.info(t("Genera un diagrama con la descripción",
                          "Generate a diagram with the description"))

    # ── TAB MÉTRICAS ──────────────────────────────────────────
    with tabs[3]:
        st.markdown(f"### 📈 {t('Métricas de calidad', 'Quality metrics')}")
        docs_m = list(st.session_state.docs_generados)
        if not docs_m and proyecto_path:
            res_dir = Path(proyecto_path) / "resultado"
            if res_dir.exists():
                for f in sorted(res_dir.glob("*.md")):
                    if not f.name.startswith('.') and '.auditoria' not in f.name:
                        try:
                            docs_m.append({"nombre": f.name,
                                           "contenido": f.read_text("utf-8")})
                        except Exception:
                            pass
        if docs_m:
            for doc in docs_m:
                contenido = doc.get('contenido', '')
                if len(contenido) > 50:
                    m = EvaluadorCalidad().evaluar(contenido, [], doc['nombre'])
                    st.markdown(f"**📄 {doc['nombre']}**")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"<div class='metric-card'><div class='metric-label'>"
                                    f"{t('Cobertura','Coverage')}</div>"
                                    f"<div class='metric-value'>{m['cobertura']}%</div></div>",
                                    unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"<div class='metric-card'><div class='metric-label'>"
                                    f"{t('Densidad','Density')}</div>"
                                    f"<div class='metric-value'>{m['densidad']}</div></div>",
                                    unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"<div class='metric-card'><div class='metric-label'>"
                                    f"{t('Legibilidad','Readability')}</div>"
                                    f"<div class='metric-value'>{m['legibilidad']}</div></div>",
                                    unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"<div class='metric-card'><div class='metric-label'>"
                                    f"{t('Palabras','Words')}</div>"
                                    f"<div class='metric-value'>{m['palabras']}</div></div>",
                                    unsafe_allow_html=True)
                    st.markdown("---")
        else:
            st.info(t("Genera documentos para ver métricas",
                      "Generate docs to see metrics"))

    # ── TAB SCRATCHPAD ────────────────────────────────────────
    with tabs[4]:
        st.markdown(f"### 📋 {t('Scratchpad del agente', 'Agent scratchpad')}")
        st.caption(t("Registro de todas las acciones ejecutadas por el agente",
                     "Log of all actions executed by the agent"))
        scratch = st.session_state.scratchpad
        if scratch:
            for entry in scratch:
                phase = entry.get('phase', '?')
                action = entry.get('action', '?')
                summary = entry.get('summary', '')[:200]
                rol = PHASE_TO_ROLE.get(phase, 'orquestador')
                emoji = ROLES.get(rol, {}).get('emoji', '🔧')
                st.markdown(
                    f"<div class='scratch-entry'>"
                    f"<strong>{emoji} [{phase.upper()}]</strong> "
                    f"<code>{action}</code>: {summary}"
                    f"</div>",
                    unsafe_allow_html=True
                )
        else:
            st.info(t("El scratchpad se llenará cuando ejecutes una tarea",
                      "The scratchpad fills up when you run a task"))


# ── COLUMNA LATERAL: ACTIVIDAD ────────────────────────────────────────
with col_side:
    st.markdown(f"**🔍 {t('Actividad', 'Activity')}**")
    if st.session_state.log:
        for entry in reversed(st.session_state.log[-15:]):
            rol = entry.get('rol', 'orquestador')
            phase = entry.get('phase', '?')
            emoji = ROLES.get(rol, {}).get('emoji', '🔧')
            st.markdown(f"<span class='role-badge role-{rol}'>{emoji} {phase}</span>",
                        unsafe_allow_html=True)
            st.caption(f"{entry.get('accion', '')}: {entry.get('resumen', '')[:100]}")
            st.markdown("---")
    else:
        st.caption(t("Sin actividad aún", "No activity yet"))

st.markdown("---")
st.caption("🧠 Agente Escritor · Pipeline Multi-Agente · RAG Semántico · Powered by DeepSeek")
