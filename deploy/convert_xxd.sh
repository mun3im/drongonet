#!/usr/bin/env bash
set -euo pipefail
TFLITE="$1"; BASE="$2"; SYM="$3"
xxd -i "$TFLITE" \
  | sed "s/unsigned char .*/alignas(8) const uint8_t ${SYM}[] = {/" \
  | sed "s/unsigned int .*/${SYM}_len;/"  \
  > "${BASE}.cc.tmp"
{ printf '#include "%s.h"\n\n' "$(basename "$BASE")"; \
  cat "${BASE}.cc.tmp"; } > "${BASE}.cc"
rm "${BASE}.cc.tmp"
cat > "${BASE}.h" <<EOF
#pragma once
#include <cstdint>
extern const uint8_t   ${SYM}[];
extern const uint32_t  ${SYM}_len;
EOF
echo "Written: ${BASE}.h  ${BASE}.cc"
