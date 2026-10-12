#!/bin/bash
# ============================================================
# iniciar.sh — Lanza la interfaz web del Agente Escritor
# ============================================================
set -e

# ─── Colores ───────────────────────────────────────────────
VERDE='\033[0;32m'
AMARILLO='\033[1;33m'
CYAN='\033[0;36m'
ROJO='\033[0;31m'
NC='\033[0m'

PUERTO=8501
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

clear
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     🧠  AGENTE ESCRITOR — Interfaz   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ─── 1. Verificar API Key ──────────────────────────────────
echo -ne "🔑 Verificando API Key... "
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo -e "${ROJO}❌ NO CONFIGURADA${NC}"
    echo ""
    echo -e "  ${AMARILLO}La variable DEEPSEEK_API_KEY no está definida.${NC}"
    echo -e "  ${AMARILLO}Asegúrate de exportarla en ~/.bashrc o antes de ejecutar:${NC}"
    echo ""
    echo -e "    ${CYAN}export DEEPSEEK_API_KEY=\"sk-tu-clave-aqui\"${NC}"
    echo ""
    exit 1
fi
echo -e "${VERDE}✅ Configurada (${#DEEPSEEK_API_KEY} chars)${NC}"

# ─── 2. Verificar archivos necesarios ──────────────────────
echo -ne "📄 Verificando app.py... "
if [ ! -f "$SCRIPT_DIR/app.py" ]; then
    echo -e "${ROJO}❌ No encontrado${NC}"
    exit 1
fi
echo -e "${VERDE}✅${NC}"

echo -ne "🐍 Verificando entorno virtual... "
if [ ! -x "$SCRIPT_DIR/env/bin/streamlit" ]; then
    echo -e "${ROJO}❌ No encontrado${NC}"
    exit 1
fi
echo -e "${VERDE}✅${NC}"

# ─── 3. Matar instancia previa en el puerto ────────────────
echo -ne "🔌 Comprobando puerto $PUERTO... "
PID_VIEJO=$(ss -tlnp 2>/dev/null | grep ":$PUERTO " | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$PID_VIEJO" ]; then
    echo -e "${AMARILLO}ocupado (PID $PID_VIEJO)${NC}"
    echo -ne "   🗑️  Terminando instancia anterior... "
    kill "$PID_VIEJO" 2>/dev/null
    sleep 1
    if kill -0 "$PID_VIEJO" 2>/dev/null; then
        kill -9 "$PID_VIEJO" 2>/dev/null
        sleep 0.5
    fi
    echo -e "${VERDE}✅ Terminada${NC}"
else
    echo -e "${VERDE}libre${NC}"
fi

# ─── 4. Lanzar Streamlit ────────────────────────────────────
echo ""
echo -ne "🚀 Lanzando Streamlit... "
"$SCRIPT_DIR/env/bin/streamlit" run app.py \
    --server.headless true \
    --server.port "$PUERTO" \
    --browser.gatherUsageStats false \
    2>&1 | while IFS= read -r linea; do
        [[ "$linea" =~ "watchdog" ]] && continue
    done &
STREAMLIT_PID=$!
echo -e "${VERDE}✅ (PID $STREAMLIT_PID)${NC}"

# ─── 5. Esperar a que el servidor responda ──────────────────
echo -ne "⏳ Esperando a que el servidor esté listo"
MAX_INTENTOS=30
for ((i=1; i<=MAX_INTENTOS; i++)); do
    sleep 1
    if curl -s -o /dev/null "http://localhost:$PUERTO" 2>/dev/null; then
        echo -e " ${VERDE}listo en ${i}s${NC}"
        break
    fi
    echo -ne "."
    if [ "$i" -eq "$MAX_INTENTOS" ]; then
        echo -e " ${ROJO}timeout${NC}"
        echo -e "${ROJO}❌ El servidor no respondió tras ${MAX_INTENTOS}s. Revisa los logs.${NC}"
        exit 1
    fi
done

# ─── 6. Abrir navegador ─────────────────────────────────────
echo ""
echo -e "${CYAN}🌐 Abriendo http://localhost:$PUERTO en tu navegador...${NC}"
xdg-open "http://localhost:$PUERTO" 2>/dev/null || true

echo ""
echo -e "${VERDE}╔══════════════════════════════════════════╗${NC}"
echo -e "${VERDE}║  ✅  Interfaz lista en puerto $PUERTO      ║${NC}"
echo -e "${VERDE}║  🛑  Pulsa Ctrl+C para detener           ║${NC}"
echo -e "${VERDE}╚══════════════════════════════════════════╝${NC}"
echo ""

# ─── 7. Esperar a que el usuario detenga ────────────────────
trap "echo ''; echo -e '${CYAN}🛑 Agente Escritor detenido. ¡Hasta pronto!${NC}'; exit 0" INT TERM
wait $STREAMLIT_PID
