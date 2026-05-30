#!/bin/bash
# SkillForge startup script
# This script is used as a fallback if supervisord is not available

# Start nginx in background
nginx &

# Start Python API server
cd /app
PYTHONPATH=/app python3 -c "
from skillforge import SkillForge
from skillforge.api import SkillForgeAPIServer

forge = SkillForge(db_path='/app/data/skillforge.db')
server = SkillForgeAPIServer(forge)
server.start()
print('SkillForge API running on http://127.0.0.1:8742')

import time
while True:
    time.sleep(3600)
" &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
