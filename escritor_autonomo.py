#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AGENTE ESCRITOR — Pipeline Multi-Agente con Scratchpad               ║
╚══════════════════════════════════════════════════════════════════════════════╝

CARACTERÍSTICAS:
  • Pipeline basado en fases: INGEST → RESEARCH → WRITE → REVIEW → REFINE → FINALIZE
  • Scratchpad persistente que preserva contexto entre iteraciones
  • RAG semántico con sentence-transformers (fallback a TF-IDF)
  • Investigación web profunda (multi-query + extracción de URLs)
  • Evaluación de calidad con LLM + heurísticas
  • Prompts dinámicos por fase y tipo de documento
  • Config unificada con migración de claves antiguas
  • Corrección de bugs (evaluar_calidad, returns duplicados)
"""

import os, sys, json, hashlib, time, shutil, argparse, sqlite3, re, textwrap, subprocess, base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

from openai import OpenAI
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Imports opcionales con degradación elegante ─────────────────────────
try: from PyPDF2 import PdfReader; PDF_OK = True
except ImportError: PDF_OK = False
try: from docx import Document; DOCX_OK = True
except ImportError: DOCX_OK = False
try: from bs4 import BeautifulSoup; HTML_OK = True
except ImportError: HTML_OK = False
try: import httpx; HTTPX_OK = True
except ImportError: HTTPX_OK = False
try: from fpdf import FPDF; FPDF_OK = True
except ImportError: FPDF_OK = False
try: import markdown; MD_OK = True
except ImportError: MD_OK = False
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_OK = True
except ImportError:
    SEMANTIC_OK = False

SCRIPT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN UNIFICADA
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    "modelo_agente":       "deepseek-chat",
    "modelo_ligero":       "deepseek-chat",
    "modelo_redactor":     "deepseek-chat",
    "modelo_verificador":  "deepseek-chat",
    "modelo_traductor":    "deepseek-chat",

    "rag_chunk_size":      2000,
    "rag_chunk_overlap":   300,
    "rag_top_k":           15,
    "rag_use_semantic":    True,

    "web_max_results":     8,
    "web_cache_horas":     24,
    "web_deep_search":     True,
    "web_max_urls_visit":  5,
    "scraping_timeout":    15,

    "max_iteraciones":     15,
    "max_chars_contexto":  140000,
    "refinamiento_max":    3,

    "idioma_principal":    "es",
    "idiomas_salida":      ["es"],

    "excluir_patrones": ["resultado/","cache/","rag_index/",".git/","__pycache__/","env/",".bak"],
    "formatos_texto": ['.txt','.md','.csv','.json','.log','.yaml','.yml','.xml','.py','.js','.ts','.sh','.cfg','.ini','.toml'],
    "formatos_rich": ['.pdf','.docx','.html','.htm'],

    "github_token": "",
    "notion_token": "",
}

# Migración de claves antiguas → nuevas
_KEY_MIGRATION = {
    "modelo_planificador": "modelo_agente",
    "modelo_busqueda": "modelo_ligero",
    "modelo_resumen": "modelo_ligero",
    "max_resultados_web": "web_max_results",
    "num_busquedas_web": "web_max_results",
    "ttl_cache_web_horas": "web_cache_horas",
    "max_chars_contexto_plan": "max_chars_contexto",
    "max_chars_contexto_redaccion": "max_chars_contexto",
}

def cargar_config():
    """Carga config.json y migra claves antiguas."""
    ruta = SCRIPT_DIR / "config.json"
    if not ruta.exists():
        return dict(CONFIG)
    with open(ruta, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Migrar claves antiguas
    migrado = {}
    for k, v in raw.items():
        if k in _KEY_MIGRATION:
            migrado[_KEY_MIGRATION[k]] = v
        else:
            migrado[k] = v
    return {**CONFIG, **migrado}


# ═══════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE IDIOMA
# ═══════════════════════════════════════════════════════════════════════════

_ES_WORDS = frozenset(['el','la','los','las','de','del','que','en','un','una','es','por','para',
    'con','se','no','al','lo','como','más','pero','si','su','sus','este','esta','entre',
    'hay','cuando','muy','sin','sobre','también','todo','porque','hasta','desde',
    'informe','análisis','datos','debe','normativa','cumplimiento','seguridad','riesgos',
    'documento','técnico','sistema','proceso','implementación','gestión','evaluación'])

_EN_WORDS = frozenset(['the','of','and','to','in','is','that','for','it','with','on','as',
    'was','are','be','this','by','not','from','or','an','but','if','can','has','its',
    'all','will','one','so','which','at','new','also','data','analysis','report',
    'must','compliance','security','risk','document','technical','system','process',
    'implementation','management','evaluation','assessment'])

def detectar_idioma(texto: str) -> str:
    """Detecta si un texto está en español o inglés."""
    palabras = texto.lower().split()
    es_count = sum(1 for p in palabras if p in _ES_WORDS)
    en_count = sum(1 for p in palabras if p in _EN_WORDS)
    if es_count > en_count * 1.5:
        return "es"
    if en_count > es_count * 1.5:
        return "en"
    return "es"  # default


# ═══════════════════════════════════════════════════════════════════════════
#  GENERADOR DE DIAGRAMAS MERMAID
# ═══════════════════════════════════════════════════════════════════════════

MERMAID_TEMPLATES = {
    "flowchart": """flowchart TD
    A[Inicio] --> B{¿Condición?}
    B -->|Sí| C[Acción 1]
    B -->|No| D[Acción 2]
    C --> E[Fin]
    D --> E""",
    "sequence": """sequenceDiagram
    participant A as Actor 1
    participant B as Actor 2
    A->>B: Mensaje
    B-->>A: Respuesta""",
    "gantt": """gantt
    title Proyecto
    dateFormat YYYY-MM-DD
    section Fase 1
    Tarea 1 :a1, 2025-01-01, 30d
    Tarea 2 :after a1, 20d""",
    "pie": """pie title Distribución
    "Categoría A" : 40
    "Categoría B" : 35
    "Categoría C" : 25""",
    "class": """classDiagram
    class Entidad {
        +atributo1
        +atributo2
        +método1()
    }
    Entidad <|-- SubEntidad""",
    "er": """erDiagram
    ENTIDAD_A ||--o{ ENTIDAD_B : tiene
    ENTIDAD_A {
        int id
        string nombre
    }
    ENTIDAD_B {
        int id
        string descripcion
    }""",
    "timeline": """timeline
    title Línea temporal
    2024 : Evento 1
    2025 : Evento 2 : Evento 3""",
    "mindmap": """mindmap
    root((Tema Principal))
        Subtema 1
            Punto A
            Punto B
        Subtema 2
            Punto C""",
}

TIPS_MERMAID = {
    "flowchart": "Usa flowchart para procesos, árboles de decisión, flujos de trabajo.",
    "sequence": "Usa sequence para interacciones entre sistemas, APIs, actores.",
    "gantt": "Usa gantt para cronogramas, plazos de implementación, fases de proyecto.",
    "pie": "Usa pie para distribuciones: presupuestos, cuotas de mercado, porcentajes.",
    "class": "Usa class para arquitectura de software, modelos de datos, jerarquías.",
    "er": "Usa er para bases de datos, entidades y relaciones.",
    "timeline": "Usa timeline para hitos históricos, evolución normativa.",
    "mindmap": "Usa mindmap para estructurar ideas, brainstorming, resúmenes visuales."
}

def generar_mermaid(client, cfg, tipo, contexto, idioma="es"):
    """Genera código Mermaid basado en el contexto de datos."""
    prompt = f"""Eres experto en diagramas Mermaid.js. Genera código Mermaid tipo '{tipo}'.

TIPO: {tipo}
DESCRIPCIÓN: {TIPS_MERMAID.get(tipo, '')}

CONTEXTO:
{contexto[:8000]}

REGLAS:
1. Extrae datos relevantes del contexto
2. Código Mermaid.js VÁLIDO y sin errores de sintaxis
3. Nombres descriptivos en {'español' if idioma == 'es' else 'inglés'}
4. Responde SOLO con el código Mermaid, sin markdown, sin explicaciones"""

    try:
        return client.chat.completions.create(
            model=cfg.get("modelo_ligero", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
    except Exception:
        return MERMAID_TEMPLATES.get(tipo, "graph TD\n    A[Sin datos] --> B[Fin]")


# ═══════════════════════════════════════════════════════════════════════════
#  TRADUCTOR EN↔ES
# ═══════════════════════════════════════════════════════════════════════════

def traducir_documento(client, cfg, texto, direccion="es->en"):
    """Traduce un documento manteniendo formato Markdown."""
    lang_map = {"es->en": ("español", "inglés"), "en->es": ("inglés", "español")}
    src, dst = lang_map.get(direccion, ("español", "inglés"))

    # Dividir documentos largos por secciones
    secciones = re.split(r'(?=^#{1,4}\s)', texto, flags=re.MULTILINE)
    if len(secciones) <= 1 or len(texto) < 5000:
        chunks = [texto]
    else:
        chunks, buf = [], ""
        for s in secciones:
            if len(buf) + len(s) < 8000:
                buf += s
            else:
                if buf: chunks.append(buf)
                buf = s
        if buf: chunks.append(buf)

    resultado = []
    for chunk in chunks:
        prompt = f"""Traduce de {src} a {dst}.
REGLAS:
1. Conserva EXACTAMENTE el formato Markdown
2. NO traduzcas nombres propios, marcas ni términos técnicos
3. Si hay código Mermaid (```mermaid...```), NO lo traduzcas
4. Devuelve SOLO la traducción

=== TEXTO ===
{chunk}"""
        try:
            trad = client.chat.completions.create(
                model=cfg.get("modelo_traductor", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            resultado.append(trad)
        except Exception as e:
            resultado.append(f"[ERROR traducción: {e}]\n{chunk}")

    return "\n\n".join(resultado)


# ═══════════════════════════════════════════════════════════════════════════
#  MOTOR RAG — Con embeddings semánticos (fallback TF-IDF)
# ═══════════════════════════════════════════════════════════════════════════

class MotorRAG:
    """Motor de Retrieval-Augmented Generation con soporte semántico."""

    def __init__(self, cfg, dir_cache):
        self.cfg = cfg
        self.dir_rag = Path(dir_cache) / "rag_index"
        self.dir_rag.mkdir(parents=True, exist_ok=True)
        self.chunks = []
        # Embeddings semánticos
        self.use_semantic = False
        self.encoder = None
        self.embeddings = None
        # Fallback TF-IDF
        self.vectorizer = None
        self.matriz = None

        if cfg.get("rag_use_semantic", True) and SEMANTIC_OK:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_semantic = True
                print("   ✅ RAG semántico activado (sentence-transformers)")
            except Exception:
                self.use_semantic = False

        self._cargar()

    def _cargar(self):
        ruta = self.dir_rag / "index.json"
        if ruta.exists():
            self.chunks = json.loads(ruta.read_text("utf-8"))
            if self.use_semantic and (self.dir_rag / "embeddings.npy").exists():
                self.embeddings = np.load(self.dir_rag / "embeddings.npy", allow_pickle=True)
            elif (self.dir_rag / "matriz.npy").exists():
                self.matriz = np.load(self.dir_rag / "matriz.npy", allow_pickle=True)
                if (self.dir_rag / "vectorizer.pkl").exists():
                    import pickle
                    with open(self.dir_rag / "vectorizer.pkl", "rb") as f:
                        self.vectorizer = pickle.load(f)

    def _guardar(self):
        (self.dir_rag / "index.json").write_text(
            json.dumps(self.chunks, indent=2, ensure_ascii=False), "utf-8")
        if self.use_semantic and self.embeddings is not None:
            np.save(self.dir_rag / "embeddings.npy", self.embeddings)
        elif self.matriz is not None:
            np.save(self.dir_rag / "matriz.npy", self.matriz)
            if self.vectorizer is not None:
                import pickle
                with open(self.dir_rag / "vectorizer.pkl", "wb") as f:
                    pickle.dump(self.vectorizer, f)

    def _chunk_texto_semantico(self, texto: str, fuente: str) -> List[Dict]:
        """Chunking inteligente: respetar estructura markdown."""
        size = self.cfg.get("rag_chunk_size", 2000)
        overlap = self.cfg.get("rag_chunk_overlap", 300)

        # Paso 1: dividir por headers markdown
        secciones = re.split(r'(^#{1,4}\s+.+$)', texto, flags=re.MULTILINE)
        bloques = []
        header_actual = ""
        for parte in secciones:
            parte = parte.strip()
            if not parte:
                continue
            if re.match(r'^#{1,4}\s+', parte):
                header_actual = parte
            else:
                # Prefixar con el header para dar contexto al chunk
                contenido = f"{header_actual}\n{parte}" if header_actual else parte
                bloques.append(contenido)

        if not bloques:
            bloques = [texto]

        # Paso 2: subdividir bloques largos por párrafos
        chunks_finales = []
        for bloque in bloques:
            if len(bloque) <= size * 1.5:
                if len(bloque.strip()) > 30:
                    chunks_finales.append(bloque)
            else:
                parrafos = [p.strip() for p in bloque.split("\n\n") if p.strip()]
                buf = ""
                for p in parrafos:
                    if len(buf) + len(p) < size:
                        buf += ("\n\n" + p) if buf else p
                    else:
                        if buf and len(buf) > 30:
                            chunks_finales.append(buf)
                        buf = p
                if buf and len(buf) > 30:
                    chunks_finales.append(buf)

        return [{"texto": c, "fuente": fuente, "chunk_id": f"{fuente}_{i}"}
                for i, c in enumerate(chunks_finales)]

    def indexar(self, contenidos: dict):
        """Indexa documentos en el motor RAG."""
        nuevos = []
        for nombre, texto in contenidos.items():
            if texto and len(texto) > 20:
                nuevos.extend(self._chunk_texto_semantico(texto, nombre))
        if not nuevos:
            return
        self.chunks = nuevos
        textos = [c["texto"] for c in self.chunks]

        if self.use_semantic and self.encoder:
            try:
                self.embeddings = self.encoder.encode(textos, show_progress_bar=False)
                print(f"   📊 {len(textos)} chunks indexados (semántico)")
            except Exception as e:
                print(f"   ⚠️ Fallback a TF-IDF: {e}")
                self.use_semantic = False
                self._indexar_tfidf(textos)
        else:
            self._indexar_tfidf(textos)

        self._guardar()

    def _indexar_tfidf(self, textos):
        """Indexación fallback con TF-IDF."""
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.matriz = self.vectorizer.fit_transform(textos)
        print(f"   📊 {len(textos)} chunks indexados (TF-IDF)")

    def consultar(self, query: str, top_k: int = None) -> tuple:
        """Consulta semántica o TF-IDF según disponibilidad."""
        if not self.chunks:
            return [], ""
        top_k = top_k or self.cfg.get("rag_top_k", 15)

        try:
            if self.use_semantic and self.embeddings is not None and self.encoder:
                q_emb = self.encoder.encode([query])
                sims = cosine_similarity(q_emb, self.embeddings)[0]
            elif self.matriz is not None and self.vectorizer:
                q_vec = self.vectorizer.transform([query])
                sims = cosine_similarity(q_vec, self.matriz)[0]
            else:
                return [], ""

            indices = np.argsort(sims)[::-1][:top_k]
            resultados = []
            for i in indices:
                if sims[i] >= 0.05:
                    resultados.append({**self.chunks[i], "score": float(sims[i])})
            texto = "\n\n".join([
                f"📄 [{r['fuente']}] (sim:{r['score']:.2f})\n{r['texto']}"
                for r in resultados
            ])
            return resultados, texto
        except Exception:
            return [], ""

    def stats(self):
        return {
            "chunks": len(self.chunks),
            "indexado": self.embeddings is not None or self.matriz is not None,
            "tipo": "semántico" if self.use_semantic else "TF-IDF"
        }


# ═══════════════════════════════════════════════════════════════════════════
#  MEMORIA SQLITE
# ═══════════════════════════════════════════════════════════════════════════

class MemoriaSQLite:
    def __init__(self, dir_cache):
        self.db_path = Path(dir_cache) / "memoria.db"
        with sqlite3.connect(self.db_path) as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS conversaciones(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proyecto TEXT, rol TEXT, contenido TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS decisiones(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proyecto TEXT, herramienta TEXT, parametros TEXT,
                    resultado_resumen TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS documentos(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proyecto TEXT, nombre TEXT, version INTEGER DEFAULT 1,
                    etapa TEXT DEFAULT 'borrador',
                    metrica_cobertura REAL, metrica_densidad REAL,
                    metrica_legibilidad REAL, idioma TEXT DEFAULT 'es',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);
            """)

    def guardar_mensaje(self, proy, rol, cont):
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO conversaciones(proyecto,rol,contenido) VALUES(?,?,?)",
                      (proy, rol, cont[:50000]))

    def guardar_decision(self, proy, herr, params, res):
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO decisiones(proyecto,herramienta,parametros,resultado_resumen) VALUES(?,?,?,?)",
                      (proy, herr, json.dumps(params)[:5000], str(res)[:2000]))

    def guardar_metricas(self, proy, nombre, cobertura, densidad, legibilidad, idioma="es", version=1):
        with sqlite3.connect(self.db_path) as c:
            c.execute("INSERT INTO documentos(proyecto,nombre,version,idioma,metrica_cobertura,metrica_densidad,metrica_legibilidad) VALUES(?,?,?,?,?,?,?)",
                      (proy, nombre, version, idioma, cobertura, densidad, legibilidad))

    def obtener_historial(self, proy, lim=30):
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("SELECT rol,contenido,timestamp FROM conversaciones WHERE proyecto=? ORDER BY id DESC LIMIT ?",
                             (proy, lim)).fetchall()
        return [{"rol": r[0], "contenido": r[1], "timestamp": r[2]} for r in reversed(rows)]


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUADOR DE CALIDAD — Heurísticas + LLM
# ═══════════════════════════════════════════════════════════════════════════

class EvaluadorCalidad:
    @staticmethod
    def cobertura_fuentes(documento: str, fuentes_chunks: list) -> float:
        if not fuentes_chunks:
            return 0.0
        doc_lower = documento.lower()
        cubiertos = 0
        for chunk in fuentes_chunks:
            texto = chunk.get("texto", "")[:200]
            frases = [f.strip().lower() for f in re.split(r'[.!?]', texto) if len(f.strip()) > 25]
            if any(f in doc_lower for f in frases):
                cubiertos += 1
        return round(cubiertos / len(fuentes_chunks) * 100, 1)

    @staticmethod
    def densidad_informativa(documento: str) -> float:
        cifras = len(re.findall(r'\d+[.,]?\d*\s*(?:%|€|USD|millones|años|meses)', documento))
        fechas = len(re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}', documento))
        nombres = len(re.findall(r'[A-ZÁÉÍÓÚ][a-záéíóú]+ [A-ZÁÉÍÓÚ][a-záéíóú]+', documento))
        palabras = max(len(documento.split()), 1)
        return round((cifras + fechas + nombres) / palabras * 1000, 1)

    @staticmethod
    def legibilidad(documento: str, idioma: str = "es") -> float:
        """Calcula legibilidad Flesch adaptada. Acepta idioma como parámetro opcional."""
        oraciones = max(len(re.split(r'[.!?]+', documento)), 1)
        palabras = max(len(documento.split()), 1)
        silabas = len(re.findall(r'[aeiouáéíóúAEIOUÁÉÍÓÚ]', documento))
        score = 206.835 - 1.015 * (palabras / oraciones) - 84.6 * (silabas / palabras)
        return max(0, min(100, round(score, 1)))

    def evaluar(self, documento, fuentes_chunks, nombre="", idioma="es"):
        return {
            "nombre": nombre,
            "idioma": idioma,
            "cobertura": self.cobertura_fuentes(documento, fuentes_chunks),
            "densidad": self.densidad_informativa(documento),
            "legibilidad": self.legibilidad(documento, idioma),
            "chars": len(documento),
            "palabras": len(documento.split()),
        }

    @staticmethod
    def evaluar_con_llm(client, cfg, documento, fuentes_texto="", idioma="es"):
        """Evaluación profunda usando el LLM como evaluador."""
        if idioma == "en":
            prompt = f"""Evaluate this document's quality. Respond ONLY with valid JSON:
{{"estructura": {{"score": 1-10, "issues": ["..."]}},
 "profundidad": {{"score": 1-10, "issues": ["..."]}},
 "precision": {{"score": 1-10, "issues": ["..."]}},
 "formato": {{"score": 1-10, "issues": ["..."]}},
 "overall": 1-10,
 "feedback_para_revision": "specific actionable feedback"}}

DOCUMENT (first 12000 chars):
{documento[:12000]}

AVAILABLE SOURCES (first 6000 chars):
{fuentes_texto[:6000] if fuentes_texto else '(no sources)'}"""
        else:
            prompt = f"""Evalúa la calidad de este documento. Responde SOLO con JSON válido:
{{"estructura": {{"score": 1-10, "issues": ["..."]}},
 "profundidad": {{"score": 1-10, "issues": ["..."]}},
 "precision": {{"score": 1-10, "issues": ["..."]}},
 "formato": {{"score": 1-10, "issues": ["..."]}},
 "overall": 1-10,
 "feedback_para_revision": "feedback específico y accionable"}}

DOCUMENTO (primeros 12000 chars):
{documento[:12000]}

FUENTES DISPONIBLES (primeros 6000 chars):
{fuentes_texto[:6000] if fuentes_texto else '(sin fuentes)'}"""

        try:
            res = client.chat.completions.create(
                model=cfg.get("modelo_verificador", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            limpio = res.strip()
            if limpio.startswith("```"):
                limpio = limpio.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(limpio)
        except Exception as e:
            return {"overall": 5, "feedback_para_revision": f"Error evaluación LLM: {e}",
                    "estructura": {"score": 5, "issues": []},
                    "profundidad": {"score": 5, "issues": []},
                    "precision": {"score": 5, "issues": []},
                    "formato": {"score": 5, "issues": []}}


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORTADOR PDF / DOCX
# ═══════════════════════════════════════════════════════════════════════════

class ExportadorPDF:
    @staticmethod
    def md_a_pdf(texto_md, ruta_salida):
        if not FPDF_OK:
            return False, "fpdf2 no instalado"
        try:
            texto_limpio = limpiar_markdown(texto_md)
            html = markdown.markdown(texto_limpio, extensions=['tables', 'fenced_code'])
            
            if HTML_OK:
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    for cell in soup.find_all(['td', 'th']):
                        cell.string = cell.get_text()
                    html = str(soup)
                except Exception:
                    pass

            pdf = FPDF()
            pdf.add_page()
            fuente = "Helvetica"
            for f in ["DejaVuSans"]:
                try:
                    pdf.add_font(f, "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
                    pdf.add_font(f, "B", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
                    pdf.add_font(f, "I", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf")
                    fuente = f
                    break
                except Exception:
                    pass
            pdf.set_font(fuente, size=10)
            pdf.set_auto_page_break(True, 15)
            pdf.write_html(html)
            pdf.output(ruta_salida)
            return True, f"PDF: {ruta_salida}"
        except Exception as e:
            return False, f"Error PDF: {e}"

    @staticmethod
    def md_a_docx(texto_md, ruta_salida):
        if not DOCX_OK:
            return False, "python-docx no instalado"
        try:
            texto_limpio = limpiar_markdown(texto_md)
            doc = Document()
            for linea in texto_limpio.split('\n'):
                linea = linea.strip()
                if not linea:
                    continue
                if linea.startswith('# ') and not linea.startswith('##'):
                    doc.add_heading(linea[2:], level=1)
                elif linea.startswith('## '):
                    doc.add_heading(linea[3:], level=2)
                elif linea.startswith('### '):
                    doc.add_heading(linea[4:], level=3)
                elif linea.startswith(('- ', '* ')):
                    doc.add_paragraph(linea[2:], style='List Bullet')
                elif linea.startswith('|'):
                    continue
                else:
                    doc.add_paragraph(linea)
            doc.save(ruta_salida)
            return True, f"DOCX: {ruta_salida}"
        except Exception as e:
            return False, f"Error DOCX: {e}"


# ═══════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def md5_file(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()

def leer_pdf(ruta):
    try:
        return "\n".join([p.extract_text() or "" for p in PdfReader(str(ruta)).pages])
    except Exception as e:
        return f"[ERROR PDF:{e}]"

def leer_docx(ruta):
    try:
        return "\n".join([p.text for p in Document(str(ruta)).paragraphs if p.text.strip()])
    except Exception as e:
        return f"[ERROR DOCX:{e}]"

def leer_html_str(html):
    try:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception:
        return html

def debe_excluirse(ruta_rel, patrones):
    return any(p in ruta_rel for p in patrones)

def scrape_url(url, timeout=15):
    if not HTTPX_OK:
        return None, "httpx no instalado"
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 AgentEscritor/8.0"})
        r.raise_for_status()
        return leer_html_str(r.text)[:80000], None
    except Exception as e:
        return None, str(e)

def limpiar_markdown(texto: str) -> str:
    """Elimina delimitadores de bloque de código markdown si el LLM los generó."""
    t = texto.strip()
    if t.startswith("```markdown"):
        t = t.split("\n", 1)[1]
    elif t.startswith("```"):
        t = t.split("\n", 1)[1]
    
    if t.endswith("```"):
        t = t.rsplit("\n", 1)[0]
    return t.strip()


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPTS POR FASE — Concisos y específicos
# ═══════════════════════════════════════════════════════════════════════════

PHASE_PROMPTS = {
    "ingest": {
        "es": """Eres un asistente documental. FASE: INGESTIÓN.
Tu única tarea: llama a leer_fuentes para leer todos los archivos del proyecto.
Actúa INMEDIATAMENTE sin hablar. Solo usa la herramienta.""",
        "en": """You are a documentation assistant. PHASE: INGESTION.
Your only task: call leer_fuentes to read all project files.
Act IMMEDIATELY without talking. Just use the tool."""
    },
    "research": {
        "es": """Eres un investigador de élite. FASE: INVESTIGACIÓN.

Tu tarea: investigar exhaustivamente sobre el OBJETIVO usando todas las fuentes disponibles.

PASOS OBLIGATORIOS:
1. consultar_fuentes — 3+ consultas diferentes sobre aspectos clave del tema
2. buscar_web — 3+ búsquedas diferentes (busca datos concretos, no conceptos genéricos)
3. visitar_url — visita las 2-3 URLs más relevantes de los resultados web
4. establecer_briefing — guarda un briefing ESTRUCTURADO con secciones:
   • DATOS CLAVE: hechos verificados, cifras, fechas
   • FUENTES: URLs y documentos consultados
   • HALLAZGOS: descubrimientos relevantes
   • BRECHAS: qué información falta

Si puedes, usa investigar_profundo para hacer todo esto en un solo paso.

NO redactes documentos. Solo investiga y guarda el briefing.""",
        "en": """You are an elite researcher. PHASE: RESEARCH.

Your task: exhaustively research the OBJECTIVE using all available sources.

MANDATORY STEPS:
1. consultar_fuentes — 3+ different queries about key topic aspects
2. buscar_web — 3+ different searches (look for concrete data, not generic concepts)
3. visitar_url — visit the 2-3 most relevant URLs from web results
4. establecer_briefing — save a STRUCTURED briefing with sections:
   • KEY DATA: verified facts, figures, dates
   • SOURCES: URLs and documents consulted
   • FINDINGS: relevant discoveries
   • GAPS: what information is missing

If available, use investigar_profundo to do all this in one step.

Do NOT write documents. Only research and save the briefing."""
    },
    "write": {
        "es": """Eres un redactor técnico de élite. FASE: REDACCIÓN.

Tu tarea: redactar el documento usando el briefing y las fuentes disponibles.

REGLAS DE REDACCIÓN:
1. Usa Markdown profesional con estructura clara (##, ###)
2. Incluye tablas comparativas donde compares datos
3. Incluye al menos 1 diagrama Mermaid (```mermaid...```)
4. Cada afirmación cita su fuente
5. NO uses viñetas genéricas sin datos concretos
6. Adapta el contenido al tipo de documento solicitado
7. Mínimo 1500 palabras para documentos técnicos

Llama a redactar_documento con instrucciones detalladas.""",
        "en": """You are an elite technical writer. PHASE: WRITING.

Your task: write the document using the briefing and available sources.

WRITING RULES:
1. Use professional Markdown with clear structure (##, ###)
2. Include comparative tables when comparing data
3. Include at least 1 Mermaid diagram (```mermaid...```)
4. Every claim cites its source
5. Do NOT use generic bullet points without concrete data
6. Adapt content to the requested document type
7. Minimum 1500 words for technical documents

Call redactar_documento with detailed instructions."""
    },
    "review": {
        "es": """Eres un editor de control de calidad. FASE: REVISIÓN.

Tu tarea: evaluar la calidad del documento generado.

PASOS:
1. verificar_documento — auditar contra fuentes
2. evaluar_calidad — obtener métricas

Si la puntuación es >= 7/10, el documento es aceptable.
Si es < 7/10, indica qué necesita mejorar.""",
        "en": """You are a quality control editor. PHASE: REVIEW.

Your task: evaluate the quality of the generated document.

STEPS:
1. verificar_documento — audit against sources
2. evaluar_calidad — get metrics

If score is >= 7/10, document is acceptable.
If < 7/10, indicate what needs improvement."""
    },
    "refine": {
        "es": """Eres un redactor técnico en fase de refinamiento. FASE: REFINAMIENTO.

El editor encontró problemas en el documento. Tu tarea:
1. Lee el feedback del editor
2. Llama a editar_documento para corregir los problemas específicos
3. NO reescribas todo — solo corrige lo que se indicó

Sé preciso y quirúrgico en tus correcciones.""",
        "en": """You are a technical writer in refinement phase. PHASE: REFINEMENT.

The editor found issues in the document. Your task:
1. Read the editor's feedback
2. Call editar_documento to fix the specific issues
3. Do NOT rewrite everything — only fix what was indicated

Be precise and surgical in your corrections."""
    },
    "finalize": {
        "es": """Eres el orquestador. FASE: FINALIZACIÓN.

El documento ha pasado la revisión. Tu tarea:
1. Llama a finalizar con un resumen del trabajo realizado
2. Si hay idiomas adicionales configurados, traduce el documento

Actúa inmediatamente.""",
        "en": """You are the orchestrator. PHASE: FINALIZATION.

The document has passed review. Your task:
1. Call finalizar with a summary of the work done
2. If additional languages are configured, translate the document

Act immediately."""
    }
}

# Tipos de documento detectables
DOCUMENT_TYPES = {
    "technical_report": {
        "keywords_es": ["informe", "análisis", "evaluación", "técnico", "report"],
        "keywords_en": ["report", "analysis", "evaluation", "technical", "assessment"],
        "focus": "depth, data, technical artifacts"
    },
    "manual": {
        "keywords_es": ["manual", "guía", "tutorial", "instrucciones", "procedimiento"],
        "keywords_en": ["manual", "guide", "tutorial", "instructions", "procedure", "how-to"],
        "focus": "step-by-step, clarity, examples"
    },
    "comparison": {
        "keywords_es": ["comparativa", "comparación", "versus", "alternativas", "vs"],
        "keywords_en": ["comparison", "versus", "alternatives", "vs", "compare"],
        "focus": "tables, matrices, pros/cons"
    },
    "proposal": {
        "keywords_es": ["propuesta", "plan", "estrategia", "implementación", "proyecto"],
        "keywords_en": ["proposal", "plan", "strategy", "implementation", "project"],
        "focus": "objectives, timeline, resources, ROI"
    },
    "summary": {
        "keywords_es": ["resumen", "síntesis", "overview", "briefing"],
        "keywords_en": ["summary", "overview", "briefing", "synopsis", "executive"],
        "focus": "conciseness, key points, actionable insights"
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  ROLES (heredados de v7, usados para identificación de fase)
# ═══════════════════════════════════════════════════════════════════════════

ROLES = {
    "orquestador":  {"nombre": "Orquestador",  "emoji": "🎯"},
    "investigador": {"nombre": "Investigador", "emoji": "🔍"},
    "redactor":     {"nombre": "Redactor",     "emoji": "✍️"},
    "editor":       {"nombre": "Editor",       "emoji": "🔎"},
    "traductor":    {"nombre": "Traductor",    "emoji": "🌐"},
}

PHASE_TO_ROLE = {
    "ingest": "orquestador",
    "research": "investigador",
    "write": "redactor",
    "review": "editor",
    "refine": "redactor",
    "finalize": "orquestador",
}


# ═══════════════════════════════════════════════════════════════════════════
#  AGENTE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class AgenteEscritor:
    """Agente de documentación con pipeline de fases y scratchpad."""

    PHASES = ["ingest", "research", "write", "review", "refine", "finalize"]

    def __init__(self, proyecto_path, cfg=None):
        self.cfg = cfg or CONFIG
        self.proyecto_path = Path(proyecto_path).resolve()
        self.dir_fuentes = self.proyecto_path
        self.dir_resultado = self.proyecto_path / "resultado"
        self.dir_cache = self.proyecto_path / "cache"
        for d in [self.dir_fuentes, self.dir_resultado, self.dir_cache]:
            d.mkdir(parents=True, exist_ok=True)

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ DEEPSEEK_API_KEY no configurada")
            sys.exit(1)
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        self.rag = MotorRAG(self.cfg, self.dir_cache)
        self.memoria = MemoriaSQLite(self.dir_cache)
        self.evaluador = EvaluadorCalidad()

        # Estado del agente
        self.messages = []
        self.iteracion = 0
        self.current_phase = "ingest"
        self.scratchpad = []          # [{action, summary, phase}]
        self.briefing = ""
        self.documentos_generados = []
        self.idioma_actual = "es"
        self.document_type = "technical_report"
        self.objetivo = ""
        self._refinement_count = 0
        self._no_action_count = 0
        self._last_review_score = None
        self._last_review_feedback = ""

        # Caché web
        self._web_cache = self._cargar_web_cache()
        self._url_cache = {}

        # Callbacks para UI
        self._callbacks = []

    def _cargar_web_cache(self):
        ruta = self.dir_cache / "web_cache.json"
        return json.loads(ruta.read_text("utf-8")) if ruta.exists() else {}

    def _guardar_web_cache(self):
        (self.dir_cache / "web_cache.json").write_text(
            json.dumps(self._web_cache, indent=2), "utf-8")

    def on_accion(self, callback):
        """Registra callback para UI: callback(iteracion, herramienta, args, resultado)."""
        self._callbacks.append(callback)

    def _notificar(self, herramienta, args, resultado):
        for cb in self._callbacks:
            try:
                cb(self.iteracion, herramienta, args, str(resultado)[:500])
            except Exception:
                pass

    # ─── LLM ─────────────────────────────────────────────────────────────

    def _llm(self, model=None, messages=None, tools=None, thinking=False, max_retries=3):
        model = model or self.cfg["modelo_agente"]
        msgs = messages or self.messages
        for attempt in range(max_retries):
            try:
                kwargs = {"model": model, "messages": msgs}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                if thinking and "reasoner" in model:
                    kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                print(f"   ⚠️ LLM error ({attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    # ─── DETECCIÓN TIPO DE DOCUMENTO ─────────────────────────────────────

    def _detect_document_type(self, objetivo: str) -> str:
        """Detecta el tipo de documento a partir del objetivo."""
        obj_lower = objetivo.lower()
        for dtype, info in DOCUMENT_TYPES.items():
            keywords = info.get("keywords_es", []) + info.get("keywords_en", [])
            if any(kw in obj_lower for kw in keywords):
                return dtype
        return "technical_report"

    # ═════════════════════════════════════════════════════════════════════
    #  HERRAMIENTAS
    # ═════════════════════════════════════════════════════════════════════

    def tool_leer_fuentes(self, args):
        formatos = set(self.cfg["formatos_texto"] + self.cfg["formatos_rich"])
        patrones = self.cfg["excluir_patrones"]
        archivos = []
        for ruta in sorted(self.dir_fuentes.rglob("*")):
            if ruta.is_file() and ruta.suffix.lower() in formatos:
                rel = str(ruta.relative_to(self.dir_fuentes))
                if not debe_excluirse(rel, patrones):
                    archivos.append(ruta)
        if not archivos:
            return "ℹ️ Sin archivos en la carpeta fuentes." if self.idioma_actual == "es" else "ℹ️ No files found in sources folder."

        contenidos = {}
        for ruta in archivos:
            nombre = str(ruta.relative_to(self.dir_fuentes))
            s = ruta.suffix.lower()
            try:
                if s == '.pdf':
                    contenidos[nombre] = leer_pdf(ruta) if PDF_OK else "[PyPDF2 not installed]"
                elif s == '.docx':
                    contenidos[nombre] = leer_docx(ruta) if DOCX_OK else "[python-docx not installed]"
                elif s in ['.html', '.htm']:
                    contenidos[nombre] = leer_html_str(ruta.read_text("utf-8")) if HTML_OK else ruta.read_text("utf-8")
                else:
                    contenidos[nombre] = ruta.read_text("utf-8")
            except Exception as e:
                contenidos[nombre] = f"[ERROR:{e}]"

        self.rag.indexar(contenidos)
        n_archivos = len(archivos)
        n_chunks = self.rag.stats()['chunks']
        tipo_rag = self.rag.stats()['tipo']

        # Detectar idioma de las fuentes
        todo = " ".join(contenidos.values())[:5000]
        self.idioma_actual = detectar_idioma(todo)

        resumen = "\n".join([f"- {n}: {len(t)} chars" for n, t in contenidos.items()])
        return f"📂 {n_archivos} archivos leídos. {n_chunks} chunks indexados ({tipo_rag}).\nIdioma: {self.idioma_actual.upper()}\n{resumen}"

    def tool_consultar_fuentes(self, args):
        query = args.get("consulta", "")
        if not query:
            return "❌ Necesito una consulta."
        _, texto = self.rag.consultar(query)
        return f"🔍 Resultados:\n\n{texto}" if texto else "🔍 Sin resultados relevantes."

    def tool_buscar_web(self, args):
        query = args.get("consulta", "")
        if not query:
            return "❌ Necesito consulta."
        ttl = timedelta(hours=self.cfg["web_cache_horas"])
        if query in self._web_cache:
            e = self._web_cache[query]
            if datetime.now() - datetime.fromisoformat(e["ts"]) < ttl:
                return f"🌐(CACHED) {e['res']}"
        try:
            with DDGS() as ddgs:
                resultados = [r for r in ddgs.text(query, max_results=self.cfg["web_max_results"])]
            texto = "\n".join([f"📌 {r['title']}\n🔗 {r['href']}\n📝 {r['body']}" for r in resultados]) or "(no results)"
        except Exception as e:
            texto = f"Error: {e}"
        self._web_cache[query] = {"res": texto, "ts": datetime.now().isoformat()}
        self._guardar_web_cache()
        return f"🌐 '{query}':\n\n{texto}"

    def tool_visitar_url(self, args):
        url = args.get("url", "")
        if not url:
            return "❌ Necesito URL."
        if url in self._url_cache:
            return f"🔗(CACHED) {url}:\n{self._url_cache[url][:3000]}"
        texto, err = scrape_url(url, self.cfg.get("scraping_timeout", 15))
        if err:
            return f"❌ Error: {err}"
        self._url_cache[url] = texto
        return f"🔗 {url}:\n{texto[:5000]}..."

    def tool_investigar_profundo(self, args):
        """Investigación profunda: multi-query + extracción de URLs + síntesis."""
        tema = args.get("tema", self.objetivo)
        if not tema:
            return "❌ Necesito un tema para investigar."

        print(f"   🔬 Investigación profunda: '{tema[:80]}'")

        # 1. Generar queries diversas con LLM
        prompt_queries = f"""Genera 4 búsquedas web diferentes para investigar: "{tema}"
Cada búsqueda debe cubrir un aspecto distinto (técnico, práctico, comparativo, problemas conocidos).
Responde SOLO con un JSON array de strings: ["query1", "query2", "query3", "query4"]"""
        try:
            res = self._llm(
                model=self.cfg["modelo_ligero"],
                messages=[{"role": "user", "content": prompt_queries}]
            ).choices[0].message.content
            limpio = res.strip()
            if limpio.startswith("```"):
                limpio = limpio.split("\n", 1)[1].rsplit("```", 1)[0]
            queries = json.loads(limpio)
            if not isinstance(queries, list):
                queries = [tema]
        except Exception:
            queries = [tema, f"{tema} best practices", f"{tema} problemas conocidos"]

        print(f"   📋 {len(queries)} queries generadas")

        # 2. Buscar todas las queries
        todos_resultados = []
        urls_encontradas = []
        for q in queries[:5]:
            resultado = self.tool_buscar_web({"consulta": q})
            todos_resultados.append(resultado)
            # Extraer URLs de los resultados
            for line in resultado.split("\n"):
                if line.startswith("🔗 "):
                    url = line[2:].strip()
                    if url.startswith("http") and url not in urls_encontradas:
                        urls_encontradas.append(url)

        # 3. Visitar URLs prometedoras
        max_urls = self.cfg.get("web_max_urls_visit", 5)
        contenidos_web = []
        for url in urls_encontradas[:max_urls]:
            print(f"   🔗 Visitando: {url[:60]}...")
            resultado_url = self.tool_visitar_url({"url": url})
            if not resultado_url.startswith("❌"):
                contenidos_web.append(resultado_url[:3000])

        # 4. Síntesis con LLM
        contexto_total = "\n\n---\n\n".join(todos_resultados + contenidos_web)
        prompt_sintesis = f"""Sintetiza toda esta investigación sobre "{tema}" en un briefing ESTRUCTURADO.

FORMATO OBLIGATORIO:
## DATOS CLAVE
(hechos verificados, cifras, fechas, versiones)

## FUENTES PRINCIPALES
(lista de URLs y documentos consultados con lo más relevante de cada uno)

## HALLAZGOS
(descubrimientos principales, información técnica relevante)

## BRECHAS
(qué información falta o no se pudo verificar)

INFORMACIÓN RECOPILADA:
{contexto_total[:30000]}"""

        try:
            briefing = self._llm(
                model=self.cfg["modelo_agente"],
                messages=[{"role": "user", "content": prompt_sintesis}]
            ).choices[0].message.content
        except Exception:
            briefing = f"## DATOS CLAVE\n{contexto_total[:5000]}"

        self.briefing = briefing
        return f"📋 Investigación profunda completada: {len(queries)} queries, {len(contenidos_web)} URLs visitadas.\n\n{briefing[:3000]}"

    def tool_redactar_documento(self, args):
        nombre = args.get("nombre_archivo", "documento.md")
        titulo = args.get("titulo", nombre)
        instrucciones = args.get("instrucciones", "")
        objetivo = args.get("objetivo", self.objetivo)
        incluir_diagrama = args.get("incluir_diagrama", "")
        idioma_salida = args.get("idioma", self.idioma_actual)

        _, ctx_rag = self.rag.consultar(f"{titulo} {instrucciones} {objetivo}", top_k=15)

        doc_type = self.document_type
        doc_focus = DOCUMENT_TYPES.get(doc_type, {}).get("focus", "depth and clarity")

        diagrama_instr = ""
        if incluir_diagrama:
            diagrama_instr = f"\n🎨 INCLUYE un diagrama Mermaid tipo '{incluir_diagrama}'. Usa bloque ```mermaid...```"

        lang = "ESPAÑOL" if idioma_salida == "es" else "ENGLISH"

        prompt = f"""Idioma: {lang}. Tipo de documento: {doc_type}. Enfoque: {doc_focus}.

🎯 OBJETIVO: {objetivo}
📄 ARCHIVO: {nombre}
📝 TÍTULO: {titulo}
📋 INSTRUCCIONES: {instrucciones}

📚 FUENTES (RAG):
{ctx_rag if ctx_rag else '(sin fuentes locales)'}

📝 BRIEFING DE INVESTIGACIÓN:
{self.briefing[:15000] if self.briefing else '(sin briefing)'}
{diagrama_instr}

REQUISITOS:
1. Markdown profesional con estructura clara
2. Tablas comparativas donde se comparen datos
3. Al menos 1 diagrama Mermaid (```mermaid...```)
4. Cada afirmación cita su fuente
5. Mínimo 1500 palabras
6. NO viñetas genéricas sin datos concretos
7. Estructura: Resumen Ejecutivo → Análisis Principal → Datos/Tablas → Diagramas → Conclusiones → Referencias"""

        contenido = self._llm(
            model=self.cfg["modelo_redactor"],
            messages=[{"role": "user", "content": prompt}],
            thinking=True
        ).choices[0].message.content

        contenido = limpiar_markdown(contenido)

        ruta = self.dir_resultado / nombre
        ruta.write_text(contenido, "utf-8")
        self.documentos_generados.append({
            "nombre": nombre, "ruta": str(ruta),
            "contenido": contenido, "idioma": idioma_salida
        })

        # Bilingüe automático si configurado
        if self.cfg.get("idiomas_salida") and len(self.cfg["idiomas_salida"]) > 1:
            actual_lang = detectar_idioma(contenido)
            otro_idioma = "en" if actual_lang == "es" else "es"
            nombre_otro = nombre.replace(".md", f"_{otro_idioma}.md")
            direccion_trad = f"{actual_lang}->{otro_idioma}"
            contenido_otro = traducir_documento(
                self.client, self.cfg, contenido, direccion_trad)
            contenido_otro = limpiar_markdown(contenido_otro)
            (self.dir_resultado / nombre_otro).write_text(contenido_otro, "utf-8")
            self.documentos_generados.append({
                "nombre": nombre_otro,
                "ruta": str(self.dir_resultado / nombre_otro),
                "contenido": contenido_otro,
                "idioma": otro_idioma
            })

        return f"✅ {nombre} ({len(contenido)} chars, {len(contenido.split())} palabras)"

    def tool_editar_documento(self, args):
        nombre = args.get("nombre_archivo", "")
        cambio = args.get("cambio", "")
        ruta = self.dir_resultado / nombre
        if not ruta.exists():
            return f"❌ {nombre} no encontrado."
        actual = ruta.read_text("utf-8")
        _, ctx = self.rag.consultar(cambio, top_k=8)

        prompt = f"""Edita SOLO lo necesario. Preserva todo lo demás.
⚠️ Mantén líneas ≤120 chars. NO trunces contenido. Preserva diagramas Mermaid.

=== DOCUMENTO ACTUAL ===
{actual}

=== CAMBIO SOLICITADO ===
{cambio}

=== CONTEXTO DE FUENTES ===
{ctx if ctx else '(sin contexto)'}

Devuelve el documento COMPLETO con las correcciones aplicadas."""

        nuevo = self._llm(
            model=self.cfg["modelo_agente"],
            messages=[{"role": "user", "content": prompt}],
            thinking=True
        ).choices[0].message.content

        nuevo = limpiar_markdown(nuevo)

        shutil.copy2(ruta, str(ruta) + ".bak")
        ruta.write_text(nuevo, "utf-8")

        # Actualizar en documentos_generados
        for doc in self.documentos_generados:
            if doc["nombre"] == nombre:
                doc["contenido"] = nuevo
                break

        return f"✅ {nombre}: {len(actual)}→{len(nuevo)} chars"

    def tool_verificar_documento(self, args):
        nombre = args.get("nombre_archivo", "")
        ruta = self.dir_resultado / nombre
        if not ruta.exists():
            return f"❌ {nombre} no encontrado."
        contenido = ruta.read_text("utf-8")
        idioma = detectar_idioma(contenido[:2000])
        _, ctx = self.rag.consultar(f"Información clave: {nombre}", top_k=20)

        prompt = f"""Audita este documento contra las fuentes. Responde JSON:
{{"faltantes":["..."],"contradicciones":["..."],"mejoras":["..."],"puntuacion":1-10}}

=== DOCUMENTO ===
{contenido[:15000]}
=== FUENTES ===
{ctx[:60000] if ctx else '(sin fuentes)'}"""

        res = self._llm(
            model=self.cfg["modelo_verificador"],
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        (self.dir_resultado / f"{nombre}.auditoria.json").write_text(res, "utf-8")

        try:
            limpio = res.strip()
            if limpio.startswith("```"):
                limpio = limpio.split("\n", 1)[1].rsplit("```", 1)[0]
            v = json.loads(limpio)
            score = v.get('puntuacion', v.get('score', 5))
            self._last_review_score = score
            self._last_review_feedback = json.dumps(v.get('mejoras', v.get('improvements', [])), ensure_ascii=False)
            return f"🔍 {nombre}: {score}/10 | Faltan:{len(v.get('faltantes', v.get('missing', [])))} | Contradicciones:{len(v.get('contradicciones', v.get('contradictions', [])))} | Mejoras:{len(v.get('mejoras', v.get('improvements', [])))}"
        except Exception:
            return f"🔍 Auditoría guardada en {nombre}.auditoria.json"

    def tool_evaluar_calidad(self, args):
        nombre = args.get("nombre_archivo", "")
        ruta = self.dir_resultado / nombre
        if not ruta.exists():
            return f"❌ {nombre} no encontrado."
        contenido = ruta.read_text("utf-8")
        idioma = detectar_idioma(contenido[:2000])

        # Evaluación heurística
        m = self.evaluador.evaluar(contenido, self.rag.chunks, nombre, idioma)

        # Evaluación con LLM
        _, ctx_fuentes = self.rag.consultar(f"Información sobre {nombre}", top_k=10)
        eval_llm = EvaluadorCalidad.evaluar_con_llm(
            self.client, self.cfg, contenido, ctx_fuentes, idioma)

        overall = eval_llm.get("overall", 5)
        self._last_review_score = overall
        self._last_review_feedback = eval_llm.get("feedback_para_revision", "")

        self.memoria.guardar_metricas(
            self.proyecto_path.name, nombre,
            m["cobertura"], m["densidad"], m["legibilidad"], idioma)

        return (f"📊 Calidad: '{nombre}'\n"
                f"   📋 Cobertura fuentes: {m['cobertura']}%\n"
                f"   📊 Densidad informativa: {m['densidad']} datos/1K palabras\n"
                f"   📖 Legibilidad: {m['legibilidad']}/100\n"
                f"   📏 {m['palabras']} palabras, {m['chars']} chars\n"
                f"   🤖 Evaluación LLM: {overall}/10\n"
                f"   💬 Feedback: {self._last_review_feedback[:200]}")

    def tool_exportar_documento(self, args):
        nombre = args.get("nombre_archivo", "")
        formato = args.get("formato", "pdf").lower()
        ruta = self.dir_resultado / nombre
        if not ruta.exists():
            return f"❌ {nombre} no encontrado."
        contenido = ruta.read_text("utf-8")
        if formato == "pdf":
            ruta_out = str(ruta).replace(".md", ".pdf")
            ok, msg = ExportadorPDF.md_a_pdf(contenido, ruta_out)
        elif formato == "docx":
            ruta_out = str(ruta).replace(".md", ".docx")
            ok, msg = ExportadorPDF.md_a_docx(contenido, ruta_out)
        else:
            return f"❌ Formato '{formato}' no soportado."
        return f"{'✅' if ok else '❌'} {msg}"

    def tool_conectar_github(self, args):
        token = self.cfg.get("github_token", "")
        repo = args.get("repo", "")
        endpoint = args.get("endpoint", "readme")
        if not token:
            return "❌ github_token no configurado."
        if not repo:
            return "❌ Especifica repo (ej: 'user/repo')."
        try:
            import requests
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
            if endpoint == "readme":
                r = requests.get(f"https://api.github.com/repos/{repo}/readme", headers=headers)
                if r.status_code == 200:
                    return f"📖 README de {repo}:\n{base64.b64decode(r.json()['content']).decode('utf-8')[:5000]}"
                return f"❌ GitHub: {r.status_code}"
            elif endpoint == "issues":
                r = requests.get(f"https://api.github.com/repos/{repo}/issues?state=open&per_page=5", headers=headers)
                if r.status_code == 200:
                    issues = [f"#{i['number']} {i['title']}" for i in r.json()]
                    return "🐛 Issues:\n" + "\n".join(issues)
                return f"❌ GitHub: {r.status_code}"
            return f"❌ Endpoint '{endpoint}' no soportado."
        except Exception as e:
            return f"❌ GitHub error: {e}"

    def tool_conectar_notion(self, args):
        token = self.cfg.get("notion_token", "")
        page_id = args.get("page_id", "")
        if not token:
            return "❌ notion_token no configurado."
        if not page_id:
            return "❌ Especifica page_id."
        try:
            import requests
            headers = {"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28"}
            r = requests.get(f"https://api.notion.com/v1/blocks/{page_id}/children?page_size=20", headers=headers)
            if r.status_code == 200:
                bloques = []
                for b in r.json().get("results", []):
                    t = b.get("type", "")
                    if t == "paragraph":
                        txt = b.get("paragraph", {}).get("rich_text", [])
                        bloques.append(" ".join([x.get("plain_text", "") for x in txt]))
                return "📄 Notion page:\n" + "\n".join(bloques)[:5000]
            return f"❌ Notion: {r.status_code}"
        except Exception as e:
            return f"❌ Notion error: {e}"

    def tool_establecer_briefing(self, args):
        self.briefing = args.get("briefing", "")
        return f"📝 Briefing guardado ({len(self.briefing)} chars)."

    def tool_generar_diagrama(self, args):
        tipo = args.get("tipo", "flowchart")
        contexto = args.get("contexto", self.briefing or "")
        if tipo not in MERMAID_TEMPLATES:
            return f"❌ Tipo desconocido: {tipo}. Disponibles: {list(MERMAID_TEMPLATES.keys())}"
        idioma = args.get("idioma", self.idioma_actual)
        codigo = generar_mermaid(self.client, self.cfg, tipo, contexto, idioma)
        nombre_diag = f"diagrama_{tipo}_{datetime.now().strftime('%H%M%S')}.md"
        ruta_diag = self.dir_resultado / nombre_diag
        ruta_diag.write_text(f"# Diagrama {tipo}\n\n```mermaid\n{codigo}\n```\n", "utf-8")
        return f"✅ Diagrama {tipo} generado: {nombre_diag}\n\n```mermaid\n{codigo}\n```"

    def tool_traducir_documento(self, args):
        nombre = args.get("nombre_archivo", "")
        direccion = args.get("direccion", "es->en")
        ruta = self.dir_resultado / nombre
        if not ruta.exists():
            return f"❌ {nombre} no encontrado."
        contenido = ruta.read_text("utf-8")
        print(f"   🌐 Traduciendo {nombre} ({direccion})...")
        trad = traducir_documento(self.client, self.cfg, contenido, direccion)
        sufijo = "_en" if direccion == "es->en" else "_es"
        nombre_trad = nombre.replace(".md", f"{sufijo}.md")
        (self.dir_resultado / nombre_trad).write_text(trad, "utf-8")
        self.documentos_generados.append({
            "nombre": nombre_trad,
            "ruta": str(self.dir_resultado / nombre_trad),
            "contenido": trad,
            "idioma": "en" if sufijo == "_en" else "es"
        })
        return f"✅ Traducido: {nombre_trad} ({len(trad)} chars)"

    def tool_finalizar(self, args):
        resumen = args.get("resumen", "Tarea completada.")
        return f"🏁 {resumen}"

    # ═════════════════════════════════════════════════════════════════════
    #  SCHEMA DE HERRAMIENTAS
    # ═════════════════════════════════════════════════════════════════════

    def herramientas_schema_por_fase(self, fase: str) -> List[Dict]:
        mapping = {
            "ingest": ["leer_fuentes"],
            "research": ["consultar_fuentes", "buscar_web", "visitar_url", "investigar_profundo", "conectar_github", "conectar_notion", "establecer_briefing"],
            "write": ["consultar_fuentes", "redactar_documento", "generar_diagrama"],
            "review": ["verificar_documento", "evaluar_calidad"],
            "refine": ["editar_documento", "consultar_fuentes"],
            "finalize": ["traducir_documento", "exportar_documento", "finalizar"],
        }
        allowed = mapping.get(fase, [])
        all_tools = self.herramientas_schema
        filtered = [t for t in all_tools if t["function"]["name"] in allowed]
        return filtered

    @property
    def herramientas_schema(self):
        return [
            {"type": "function", "function": {"name": "leer_fuentes", "description": "Lee todos los archivos fuente del proyecto e indexa en RAG.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "consultar_fuentes", "description": "Búsqueda semántica en las fuentes indexadas.", "parameters": {"type": "object", "properties": {"consulta": {"type": "string"}}, "required": ["consulta"]}}},
            {"type": "function", "function": {"name": "buscar_web", "description": "Buscar en internet con DuckDuckGo.", "parameters": {"type": "object", "properties": {"consulta": {"type": "string"}}, "required": ["consulta"]}}},
            {"type": "function", "function": {"name": "visitar_url", "description": "Visitar y extraer contenido de una URL.", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
            {"type": "function", "function": {"name": "investigar_profundo", "description": "Investigación profunda: genera múltiples queries, busca web, visita URLs y sintetiza un briefing estructurado.", "parameters": {"type": "object", "properties": {"tema": {"type": "string", "description": "Tema a investigar"}}, "required": ["tema"]}}},
            {"type": "function", "function": {"name": "redactar_documento", "description": "Redactar documento Markdown completo con diagramas Mermaid opcionales.", "parameters": {"type": "object", "properties": {"nombre_archivo": {"type": "string"}, "titulo": {"type": "string"}, "instrucciones": {"type": "string"}, "objetivo": {"type": "string"}, "incluir_diagrama": {"type": "string"}, "idioma": {"type": "string", "description": "es o en"}}, "required": ["nombre_archivo", "titulo", "instrucciones"]}}},
            {"type": "function", "function": {"name": "editar_documento", "description": "Editar documento existente con cambios específicos.", "parameters": {"type": "object", "properties": {"nombre_archivo": {"type": "string"}, "cambio": {"type": "string"}}, "required": ["nombre_archivo", "cambio"]}}},
            {"type": "function", "function": {"name": "verificar_documento", "description": "Auditar documento contra fuentes. Genera puntuación y feedback.", "parameters": {"type": "object", "properties": {"nombre_archivo": {"type": "string"}}, "required": ["nombre_archivo"]}}},
            {"type": "function", "function": {"name": "evaluar_calidad", "description": "Evaluar métricas de calidad del documento (heurísticas + LLM).", "parameters": {"type": "object", "properties": {"nombre_archivo": {"type": "string"}}, "required": ["nombre_archivo"]}}},
            {"type": "function", "function": {"name": "exportar_documento", "description": "Exportar a PDF o DOCX.", "parameters": {"type": "object", "properties": {"nombre_archivo": {"type": "string"}, "formato": {"type": "string"}}, "required": ["nombre_archivo", "formato"]}}},
            {"type": "function", "function": {"name": "conectar_github", "description": "Leer README o issues de GitHub.", "parameters": {"type": "object", "properties": {"repo": {"type": "string"}, "endpoint": {"type": "string"}}, "required": ["repo"]}}},
            {"type": "function", "function": {"name": "conectar_notion", "description": "Leer página de Notion.", "parameters": {"type": "object", "properties": {"page_id": {"type": "string"}}, "required": ["page_id"]}}},
            {"type": "function", "function": {"name": "establecer_briefing", "description": "Guardar briefing estructurado de investigación.", "parameters": {"type": "object", "properties": {"briefing": {"type": "string"}}, "required": ["briefing"]}}},
            {"type": "function", "function": {"name": "generar_diagrama", "description": "Generar diagrama Mermaid (flowchart, sequence, gantt, pie, class, er, timeline, mindmap).", "parameters": {"type": "object", "properties": {"tipo": {"type": "string", "description": "flowchart, sequence, gantt, pie, class, er, timeline, mindmap"}, "contexto": {"type": "string"}, "idioma": {"type": "string"}}, "required": ["tipo"]}}},
            {"type": "function", "function": {"name": "traducir_documento", "description": "Traducir documento EN↔ES preservando Markdown.", "parameters": {"type": "object", "properties": {"nombre_archivo": {"type": "string"}, "direccion": {"type": "string", "description": "es->en o en->es"}}, "required": ["nombre_archivo", "direccion"]}}},
            {"type": "function", "function": {"name": "finalizar", "description": "Marcar tarea como completada con resumen.", "parameters": {"type": "object", "properties": {"resumen": {"type": "string"}}, "required": ["resumen"]}}},
        ]

    def ejecutar_herramienta(self, nombre, args):
        mapa = {
            "leer_fuentes": self.tool_leer_fuentes,
            "consultar_fuentes": self.tool_consultar_fuentes,
            "buscar_web": self.tool_buscar_web,
            "visitar_url": self.tool_visitar_url,
            "investigar_profundo": self.tool_investigar_profundo,
            "redactar_documento": self.tool_redactar_documento,
            "editar_documento": self.tool_editar_documento,
            "verificar_documento": self.tool_verificar_documento,
            "evaluar_calidad": self.tool_evaluar_calidad,
            "exportar_documento": self.tool_exportar_documento,
            "conectar_github": self.tool_conectar_github,
            "conectar_notion": self.tool_conectar_notion,
            "establecer_briefing": self.tool_establecer_briefing,
            "generar_diagrama": self.tool_generar_diagrama,
            "traducir_documento": self.tool_traducir_documento,
            "finalizar": self.tool_finalizar,
        }
        func = mapa.get(nombre)
        if not func:
            return f"❌ Herramienta '{nombre}' no encontrada."
        try:
            resultado = func(args)
            self._notificar(nombre, args, resultado)
            return resultado
        except Exception as e:
            return f"❌ Error en {nombre}: {e}"

    # ═════════════════════════════════════════════════════════════════════
    #  PIPELINE DE FASES
    # ═════════════════════════════════════════════════════════════════════

    def _build_phase_system(self, phase: str, objetivo: str) -> str:
        """Construye el system prompt para una fase específica."""
        lang_key = self.idioma_actual if self.idioma_actual in ("es", "en") else "es"
        phase_prompt = PHASE_PROMPTS.get(phase, {}).get(lang_key, "Actúa según la fase actual.")

        rol_key = PHASE_TO_ROLE.get(phase, "orquestador")
        rol = ROLES.get(rol_key, ROLES["orquestador"])

        # Formatear scratchpad
        scratchpad_text = self._format_scratchpad()

        return f"""{rol['emoji']} ROL: {rol['nombre']} | FASE: {phase.upper()}

{phase_prompt}

🎯 OBJETIVO: {objetivo}
📁 PROYECTO: {self.proyecto_path.name}
📂 RESULTADOS: {self.dir_resultado}
🌐 Idioma: {self.idioma_actual.upper()}
📋 Tipo documento: {self.document_type}

{'📝 PROGRESO:' + chr(10) + scratchpad_text if scratchpad_text else ''}

{'💬 FEEDBACK EDITOR: ' + self._last_review_feedback[:500] if phase == 'refine' and self._last_review_feedback else ''}

⚠️ REGLAS: NO hables, NO saludes, NO preguntes. USA HERRAMIENTAS inmediatamente."""

    def _format_scratchpad(self) -> str:
        """Formatea el scratchpad como texto resumido."""
        if not self.scratchpad:
            return ""
        lines = []
        for entry in self.scratchpad:
            phase = entry.get("phase", "?")
            action = entry.get("action", "?")
            summary = entry.get("summary", "")[:150]
            lines.append(f"  ✅ [{phase}] {action}: {summary}")
        return "\n".join(lines[-15:])  # Últimas 15 acciones

    def _phase_instruction(self, phase: str) -> str:
        """Instrucción de usuario para iniciar una fase."""
        instructions = {
            "ingest": "🚀 Comienza: lee todas las fuentes del proyecto.",
            "research": f"🔍 Investiga sobre: {self.objetivo}",
            "write": f"✍️ Redacta el documento sobre: {self.objetivo}",
            "review": f"🔎 Revisa el documento generado: {self.documentos_generados[-1]['nombre'] if self.documentos_generados else 'documento.md'}",
            "refine": f"🔧 Corrige los problemas: {self._last_review_feedback[:300]}",
            "finalize": "🏁 Finaliza el trabajo y genera el resumen.",
        }
        return instructions.get(phase, f"Ejecuta la fase: {phase}")

    def _check_advance_phase(self):
        """Verifica si la fase actual se completó y avanza."""
        actions = [s["action"] for s in self.scratchpad]
        phase = self.current_phase

        if phase == "ingest" and "leer_fuentes" in actions:
            self.current_phase = "research"
            print(f"   ⏭️ Avanzando a fase: RESEARCH")
            self._reset_phase_messages()

        elif phase == "research":
            has_briefing = "establecer_briefing" in actions or "investigar_profundo" in actions
            has_search = sum(1 for a in actions if a == "buscar_web") >= 2
            if has_briefing or has_search:
                self.current_phase = "write"
                print(f"   ⏭️ Avanzando a fase: WRITE")
                self._reset_phase_messages()

        elif phase == "write" and "redactar_documento" in actions:
            self.current_phase = "review"
            print(f"   ⏭️ Avanzando a fase: REVIEW")
            self._reset_phase_messages()

        elif phase == "review":
            has_review = "verificar_documento" in actions or "evaluar_calidad" in actions
            if has_review and self._last_review_score is not None:
                if self._last_review_score >= 7:
                    self.current_phase = "finalize"
                    print(f"   ⏭️ Score {self._last_review_score}/10 ≥ 7 → FINALIZE")
                else:
                    if self._refinement_count < self.cfg.get("refinamiento_max", 3):
                        self.current_phase = "refine"
                        print(f"   ⏭️ Score {self._last_review_score}/10 < 7 → REFINE (intento {self._refinement_count + 1})")
                    else:
                        self.current_phase = "finalize"
                        print(f"   ⏭️ Máx refinamientos alcanzado → FINALIZE")
                self._reset_phase_messages()

        elif phase == "refine" and "editar_documento" in actions:
            self._refinement_count += 1
            # Limpiar acciones de review para que pueda re-evaluarse
            self.scratchpad = [s for s in self.scratchpad
                               if s["phase"] not in ("review",)]
            self._last_review_score = None
            self.current_phase = "review"
            print(f"   ⏭️ Refinamiento hecho → REVIEW")
            self._reset_phase_messages()

    def _reset_phase_messages(self):
        """Resetea mensajes para nueva fase (el scratchpad mantiene el contexto)."""
        self.messages = []
        self._no_action_count = 0

    def _force_advance(self):
        """Fuerza avance cuando el agente no usa herramientas."""
        phase = self.current_phase

        if phase == "ingest":
            # Forzar lectura de fuentes
            self.messages.append({"role": "user",
                "content": "USA la herramienta leer_fuentes AHORA. No hables."})
        elif phase == "research":
            # Si ya buscó algo, avanzar a write
            actions = [s["action"] for s in self.scratchpad]
            if any(a in actions for a in ["buscar_web", "consultar_fuentes", "investigar_profundo"]):
                self.current_phase = "write"
                self._reset_phase_messages()
            else:
                self.messages.append({"role": "user",
                    "content": "USA buscar_web o investigar_profundo AHORA."})
        elif phase == "write":
            self.messages.append({"role": "user",
                "content": "USA redactar_documento AHORA."})
        elif phase == "review":
            if self.documentos_generados:
                self.messages.append({"role": "user",
                    "content": f"USA evaluar_calidad con nombre_archivo='{self.documentos_generados[-1]['nombre']}' AHORA."})
            else:
                self.current_phase = "finalize"
                self._reset_phase_messages()
        elif phase == "refine":
            if self.documentos_generados:
                self.messages.append({"role": "user",
                    "content": f"USA editar_documento para corregir: {self._last_review_feedback[:200]}"})
            else:
                self.current_phase = "finalize"
                self._reset_phase_messages()
        elif phase == "finalize":
            # Auto-finalizar
            self.tool_finalizar({"resumen": f"Documento completado: {len(self.documentos_generados)} docs generados."})
            self.current_phase = "done"

    def _compress_context(self):
        """Comprime mensajes antiguos para liberar espacio de contexto."""
        if len(self.messages) <= 8:
            return
        # Mantener system (0) + últimos 6 mensajes
        system = self.messages[0]
        to_compress = self.messages[1:-6]
        recent = self.messages[-6:]

        # Resumir mensajes antiguos
        resumen_texto = []
        for m in to_compress:
            role = m.get("role", "?")
            content = m.get("content", "")[:200]
            if content:
                resumen_texto.append(f"[{role}]: {content}")

        resumen = "CONTEXTO ANTERIOR (resumido):\n" + "\n".join(resumen_texto[-10:])
        self.messages = [system, {"role": "user", "content": resumen}] + recent

    def _finalize_output(self) -> str:
        """Genera el output final con métricas de los documentos."""
        print(f"\n{'='*60}")
        print(f"🏁 Completado en {self.iteracion} iteraciones")
        print(f"{'='*60}")

        for doc in self.documentos_generados:
            print(f"\n📊 {doc['nombre']}:")
            try:
                print(self.tool_evaluar_calidad({"nombre_archivo": doc['nombre']}))
            except Exception:
                print(f"   ({len(doc.get('contenido', ''))} chars)")

        n_docs = len(self.documentos_generados)
        nombres = ", ".join([d["nombre"] for d in self.documentos_generados])
        return f"🏁 Completado: {n_docs} documento(s) generado(s): {nombres}"

    # ═════════════════════════════════════════════════════════════════════
    #  BUCLE PRINCIPAL — Pipeline de fases con scratchpad
    # ═════════════════════════════════════════════════════════════════════

    def ejecutar(self, objetivo, max_iteraciones=None):
        """Ejecuta el pipeline completo de generación de documentación."""
        max_iter = max_iteraciones or self.cfg["max_iteraciones"]
        self.objetivo = objetivo
        self.scratchpad = []
        self.documentos_generados = []
        self.iteracion = 0
        self.current_phase = "ingest"
        self._refinement_count = 0
        self._no_action_count = 0
        self._last_review_score = None
        self._last_review_feedback = ""
        self.idioma_actual = detectar_idioma(objetivo)
        self.document_type = self._detect_document_type(objetivo)
        self.briefing = ""
        self.messages = []
        proyecto = self.proyecto_path.name

        print(f"\n{'='*60}")
        print(f"🤖 AGENTE ESCRITOR — {self.idioma_actual.upper()} — Pipeline de Fases")
        print(f"📋 Tipo documento: {self.document_type}")
        print(f"🎯 Objetivo: {objetivo[:80]}...")
        print(f"{'='*60}")

        while self.iteracion < max_iter and self.current_phase != "done":
            self.iteracion += 1

            # Construir system prompt para la fase actual
            system = self._build_phase_system(self.current_phase, objetivo)

            # Si no hay mensajes o cambió la fase, iniciar conversación fresca
            if not self.messages:
                self.messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": self._phase_instruction(self.current_phase)}
                ]
            else:
                # Actualizar system prompt
                self.messages[0] = {"role": "system", "content": system}

            # Comprimir contexto si es necesario
            if len(self.messages) > 25:
                self._compress_context()

            rol = PHASE_TO_ROLE.get(self.current_phase, "orquestador")
            rol_info = ROLES.get(rol, ROLES["orquestador"])
            print(f"\n🔄 [{self.iteracion}/{max_iter}] {rol_info['emoji']} {rol_info['nombre']} — Fase: {self.current_phase.upper()}")

            try:
                response = self._llm(
                    model=self.cfg["modelo_agente"],
                    messages=self.messages,
                    tools=self.herramientas_schema_por_fase(self.current_phase)
                )
            except Exception as e:
                print(f"   ❌ Error LLM: {e}")
                continue

            msg = response.choices[0].message

            if msg.tool_calls:
                self._no_action_count = 0
                for tc in msg.tool_calls:
                    nombre = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    print(f"   🔧 {nombre}({json.dumps(args, ensure_ascii=False)[:100]})")

                    resultado = self.ejecutar_herramienta(nombre, args)

                    # Mostrar resultado resumido
                    for l in str(resultado).split('\n')[:3]:
                        print(f"      {l[:120]}")
                    if len(str(resultado).split('\n')) > 3:
                        print(f"      ...")

                    # Agregar al scratchpad
                    self.scratchpad.append({
                        "action": nombre,
                        "summary": str(resultado)[:300],
                        "phase": self.current_phase
                    })

                    self.memoria.guardar_decision(proyecto, nombre, args, str(resultado)[:500])

                    # Agregar a mensajes
                    self.messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [{"id": tc.id, "type": "function",
                                        "function": {"name": nombre, "arguments": tc.function.arguments}}]
                    })
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(resultado)[:8000]
                    })

                    self._notificar(nombre, args, resultado)

                    # Verificar finalización
                    if nombre == "finalizar":
                        self.current_phase = "done"
                        return self._finalize_output()

                # Verificar si avanzar de fase
                self._check_advance_phase()

            else:
                # Respuesta de texto sin herramientas
                texto = msg.content or ""
                self._no_action_count += 1
                print(f"💬 {texto[:200]}{'...' if len(texto) > 200 else ''}")
                self.messages.append({"role": "assistant", "content": texto})
                self.memoria.guardar_mensaje(proyecto, "assistant", texto)

                if self._no_action_count >= 2:
                    self._force_advance()

        # Si agotó iteraciones
        if self.current_phase != "done":
            print(f"\n⚠️ Límite de iteraciones ({max_iter}). Finalizando...")
            if self.documentos_generados:
                return self._finalize_output()
            return "⚠️ No se completó el proceso."

        return self._finalize_output()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Agente Escritor — Pipeline Multi-Agente")
    parser.add_argument("objetivo", nargs="?", help="Objetivo de documentación")
    parser.add_argument("--proyecto", "-p", required=True, help="Ruta al directorio del proyecto")
    parser.add_argument("--iteraciones", type=int, default=15, help="Máximo de iteraciones")
    parser.add_argument("--limpiar-cache", action="store_true", help="Limpiar caché antes de ejecutar")
    args = parser.parse_args()

    cfg = cargar_config()
    objetivo = args.objetivo or input("🎯 Objetivo: ").strip()
    if not objetivo:
        print("❌ Necesito un objetivo.")
        return

    if args.limpiar_cache:
        d = Path(args.proyecto) / "cache"
        if d.exists():
            shutil.rmtree(d)
            print(f"🧹 Caché limpiado: {d}")

    agente = AgenteEscritor(args.proyecto, cfg)
    agente.ejecutar(objetivo, max_iteraciones=args.iteraciones)


if __name__ == "__main__":
    main()
