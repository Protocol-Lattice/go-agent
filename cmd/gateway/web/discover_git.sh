#!/bin/sh
cat <<'EOF'
{"version":"1.0","tools":[{"name":"exec","description":"Run a git command in the gateway working directory.","inputs":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]},"outputs":{"type":"object","properties":{"stdout":{"type":"string"},"stderr":{"type":"string"},"exit_code":{"type":"integer"}}}}]}
EOF
