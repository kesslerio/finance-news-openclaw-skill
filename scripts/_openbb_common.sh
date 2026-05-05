#!/usr/bin/env bash
set -euo pipefail

resolve_openbb_python() {
    if [[ -n "${OPENBB_PYTHON:-}" ]]; then
        if [[ -x "${OPENBB_PYTHON}" ]]; then
            printf "%s\n" "${OPENBB_PYTHON}"
            return 0
        fi
        echo "OPENBB_PYTHON is set but not executable: ${OPENBB_PYTHON}" >&2
        return 1
    fi

    local venv_python="${HOME}/.local/venvs/openbb/bin/python3"
    if [[ -x "${venv_python}" ]]; then
        printf "%s\n" "${venv_python}"
        return 0
    fi

    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi

    echo "No usable python3 found for OpenBB wrappers." >&2
    return 1
}

resolve_openbb_provider() {
    local explicit="${1:-}"
    local fallback="${2:-yfinance}"
    if [[ -n "${explicit}" ]]; then
        printf "%s\n" "${explicit}"
    elif [[ -n "${OPENBB_DEFAULT_PROVIDER:-}" ]]; then
        printf "%s\n" "${OPENBB_DEFAULT_PROVIDER}"
    else
        printf "%s\n" "${fallback}"
    fi
}

run_openbb_python() {
    local script=""
    script="$(cat)"
    local py=""
    py="$(resolve_openbb_python)" || return 1
    "${py}" 2>&1 <<<"${script}"
}
