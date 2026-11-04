#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
bash CN/build.sh
bash EN/build.sh
