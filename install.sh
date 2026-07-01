#!/usr/bin/env bash
#
# amenity-ai installer
# Usage: curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/install.sh | sh
#
# Installs to:
#   ~/.local/share/amenity-stuff/venv/  (isolated Python environment)
#   ~/.local/bin/amenity-ai          (launcher script)
#
set -e

REPO_URL="git+https://github.com/elmisi/amenity-ai.git"
APP_NAME="amenity-ai"
INSTALL_DIR="$HOME/.local/share/amenity-stuff"
BIN_DIR="$HOME/.local/bin"
MIN_PYTHON_VERSION="3.10"

# Colors (disabled if not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    BOLD=''
    NC=''
fi

info()    { printf "${BLUE}ℹ${NC} %s\n" "$1"; }
success() { printf "${GREEN}✓${NC} %s\n" "$1"; }
warn()    { printf "${YELLOW}⚠${NC} %s\n" "$1"; }
error()   { printf "${RED}✗${NC} %s\n" "$1"; }
header()  { printf "\n${BOLD}%s${NC}\n" "$1"; }

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

# Compare version strings (returns 0 if $1 >= $2)
version_gte() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Get Python version string (e.g., "3.11")
python_version() {
    "$1" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null
}

# Find a suitable Python >= 3.10
find_python() {
    for cmd in python3 python python3.13 python3.12 python3.11 python3.10; do
        if has_cmd "$cmd"; then
            ver=$(python_version "$cmd")
            if [ -n "$ver" ] && version_gte "$ver" "$MIN_PYTHON_VERSION"; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# Check if Ollama is running
ollama_is_running() {
    if has_cmd ollama; then
        ollama list >/dev/null 2>&1
        return $?
    fi
    return 1
}

# Get list of installed Ollama models
ollama_models() {
    ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | grep -v '^$' || true
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

header "Installing $APP_NAME"

OS=$(detect_os)
info "Detected OS: $OS"

if [ "$OS" = "unknown" ]; then
    error "Unsupported operating system"
    exit 1
fi

# --- Python check ---
header "Checking Python"

PYTHON_CMD=$(find_python) || true
if [ -z "$PYTHON_CMD" ]; then
    error "Python >= $MIN_PYTHON_VERSION not found"
    printf "\n"
    if [ "$OS" = "linux" ]; then
        info "Install Python with your package manager, e.g.:"
        printf "  sudo apt install python3\n"
    else
        info "Install Python from https://www.python.org/downloads/ or via Homebrew:"
        printf "  brew install python@3.11\n"
    fi
    exit 1
fi

PYTHON_VER=$(python_version "$PYTHON_CMD")
success "Found $PYTHON_CMD ($PYTHON_VER)"

# --- Check for existing installation ---
if [ -d "$INSTALL_DIR" ]; then
    warn "Existing installation found at $INSTALL_DIR"
    printf "Reinstall? [y/N] "
    read -r answer
    case "$answer" in
        [Yy]*)
            info "Removing existing installation..."
            rm -rf "$INSTALL_DIR"
            ;;
        *)
            info "Installation cancelled"
            exit 0
            ;;
    esac
fi

# --- Create venv ---
header "Creating isolated environment"

mkdir -p "$INSTALL_DIR"
info "Creating venv at $INSTALL_DIR/venv..."

if "$PYTHON_CMD" -m venv "$INSTALL_DIR/venv"; then
    success "Created venv"
else
    error "Failed to create venv"
    rm -rf "$INSTALL_DIR"
    exit 1
fi

# --- Install package ---
header "Installing $APP_NAME"

info "Installing from GitHub..."
if "$INSTALL_DIR/venv/bin/pip" install --quiet "$REPO_URL"; then
    success "Installed $APP_NAME"
else
    error "pip install failed"
    rm -rf "$INSTALL_DIR"
    exit 1
fi

# --- Create launcher script ---
header "Creating launcher"

mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/$APP_NAME" << 'LAUNCHER'
#!/usr/bin/env bash
exec "$HOME/.local/share/amenity-stuff/venv/bin/amenity-ai" "$@"
LAUNCHER

chmod +x "$BIN_DIR/$APP_NAME"
success "Created $BIN_DIR/$APP_NAME"

# --- Verify installation ---
header "Verifying installation"

export PATH="$BIN_DIR:$PATH"

if has_cmd "$APP_NAME"; then
    success "$APP_NAME is available"
else
    warn "$APP_NAME not found in PATH"
    info "Add ~/.local/bin to your PATH:"
    printf "  export PATH=\"\$HOME/.local/bin:\$PATH\"\n"
fi

# --- Ollama check ---
header "Checking Ollama (LLM provider)"

if ! has_cmd ollama; then
    warn "Ollama not found"
    printf "\n"
    info "Ollama is required to run $APP_NAME."
    info "Install Ollama:"
    printf "  ${BOLD}curl -fsSL https://ollama.com/install.sh | sh${NC}\n"
    printf "\n"
elif ! ollama_is_running; then
    warn "Ollama is installed but not running"
    printf "\n"
    info "Start Ollama with:"
    printf "  ${BOLD}ollama serve${NC}\n"
    printf "\n"
    info "(or run it in the background / as a service)\n"
else
    success "Ollama is installed and running"

    # Check models
    MODELS=$(ollama_models)
    if [ -z "$MODELS" ]; then
        warn "No models installed"
        printf "\n"
    else
        info "Installed models:"
        printf "%s\n" "$MODELS" | sed 's/^/  /'
        printf "\n"
    fi
fi

# Model suggestions
info "Suggested models (adjust based on your hardware):"
printf "  ${BOLD}ollama pull qwen2.5:3b-instruct${NC}  # text (lightweight)\n"
printf "  ${BOLD}ollama pull moondream${NC}        # vision (for images)\n"
printf "\n"
info "For more capable hardware, consider larger models:"
printf "  ollama pull qwen2.5:7b        # better quality\n"
printf "  ollama pull llava             # alternative vision model\n"

# --- Optional dependencies ---
header "Optional system dependencies"

printf "These improve extraction quality but are not required:\n\n"

# Tesseract
if has_cmd tesseract; then
    success "tesseract (OCR) - installed"
else
    info "tesseract (OCR for scanned PDFs/images):"
    if [ "$OS" = "linux" ]; then
        printf "  sudo apt install tesseract-ocr tesseract-ocr-ita\n"
    else
        printf "  brew install tesseract tesseract-lang\n"
    fi
fi

# LibreOffice
if has_cmd libreoffice || has_cmd soffice; then
    success "libreoffice (.doc/.xls extraction) - installed"
else
    info "libreoffice (.doc/.xls extraction):"
    if [ "$OS" = "linux" ]; then
        printf "  sudo apt install libreoffice\n"
    else
        printf "  brew install --cask libreoffice\n"
    fi
fi

# unrtf
if has_cmd unrtf; then
    success "unrtf (.rtf extraction) - installed"
else
    info "unrtf (.rtf extraction):"
    if [ "$OS" = "linux" ]; then
        printf "  sudo apt install unrtf\n"
    else
        printf "  brew install unrtf\n"
    fi
fi

# --- Summary ---
header "Installation complete"

printf "Installed to:\n"
printf "  ${BOLD}$INSTALL_DIR/venv/${NC}  (Python environment)\n"
printf "  ${BOLD}$BIN_DIR/$APP_NAME${NC}  (launcher)\n"
printf "\n"
printf "Run the application:\n"
printf "  ${BOLD}$APP_NAME${NC}\n"
printf "\n"
printf "Or specify source/archive folders:\n"
printf "  ${BOLD}$APP_NAME --source /path/to/files --archive /path/to/archive${NC}\n"
printf "\n"
printf "Press F2 in the TUI to configure settings.\n"
printf "\n"
printf "To uninstall:\n"
printf "  ${BOLD}curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/uninstall.sh | sh${NC}\n"
