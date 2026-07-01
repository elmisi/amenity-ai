#!/usr/bin/env bash
#
# amenity-ai uninstaller
# Usage: curl -sSL https://raw.githubusercontent.com/elmisi/amenity-ai/main/uninstall.sh | sh
#
set -e

APP_NAME="amenity-ai"
INSTALL_DIR="$HOME/.local/share/amenity-stuff"
BIN_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.config/amenity-stuff"

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
# Main
# -----------------------------------------------------------------------------

header "Uninstalling $APP_NAME"

FOUND_SOMETHING=false

# Remove launcher script
if [ -f "$BIN_DIR/$APP_NAME" ]; then
    rm -f "$BIN_DIR/$APP_NAME"
    success "Removed $BIN_DIR/$APP_NAME"
    FOUND_SOMETHING=true
fi

# Remove venv directory
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    success "Removed $INSTALL_DIR"
    FOUND_SOMETHING=true
fi

if [ "$FOUND_SOMETHING" = false ]; then
    warn "$APP_NAME installation not found"
fi

# Config directory
if [ -d "$CONFIG_DIR" ]; then
    printf "\n"
    printf "Remove configuration directory?\n"
    printf "  ${YELLOW}%s${NC}\n" "$CONFIG_DIR"
    printf "  (contains settings, taxonomy, etc.)\n"
    printf "Remove? [y/N] "
    read -r answer
    case "$answer" in
        [Yy]*)
            rm -rf "$CONFIG_DIR"
            success "Removed $CONFIG_DIR"
            ;;
        *)
            info "Kept $CONFIG_DIR"
            ;;
    esac
fi

printf "\n"
info "Note: Cache directories (.amenity-stuff/) in source folders are not removed."
info "      Remove them manually if needed."

printf "\n"
success "Uninstall complete"
