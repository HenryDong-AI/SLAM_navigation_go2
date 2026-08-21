#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${project_root}/scripts/env.sh"

save_timeout="${GO2_SAVE_TIMEOUT_SEC:-30}"
case "${save_timeout}" in
    ""|*[!0-9]*)
        echo "GO2_SAVE_TIMEOUT_SEC must be a positive integer" >&2
        exit 2
        ;;
esac
if ((save_timeout < 1)); then
    echo "GO2_SAVE_TIMEOUT_SEC must be at least 1" >&2
    exit 2
fi

call_save_service() {
    local service_name="$1"
    local required="$2"
    local service_type
    if ! service_type="$(timeout --foreground --signal=INT --kill-after=1s \
        5s ros2 service type "${service_name}" 2>/dev/null)"; then
        if [[ "${required}" == "true" ]]; then
            echo "Required save service is unavailable: ${service_name}" >&2
            return 1
        fi
        echo "Optional semantic save service is not running; skipped."
        return 0
    fi
    if [[ "${service_type}" != "std_srvs/srv/Trigger" ]]; then
        echo "Unexpected type for ${service_name}: ${service_type}" >&2
        return 1
    fi
    if ! timeout --foreground --signal=INT --kill-after=3s \
        "${save_timeout}s" ros2 service call \
        "${service_name}" std_srvs/srv/Trigger "{}"; then
        echo "Timed out or failed saving through ${service_name}" >&2
        return 1
    fi
}

call_save_service /go2/map/save true
call_save_service /go2/semantic/save false
