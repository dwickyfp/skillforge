#!/usr/bin/env python3
"""SkillForge API server entry point for Docker."""

from skillforge import SkillForge
from skillforge.api import SkillForgeAPIServer

forge = SkillForge(db_path='/app/data/skillforge.db')
server = SkillForgeAPIServer(forge)
server.start()
print('SkillForge API running on http://127.0.0.1:8742')

import time
while True:
    time.sleep(3600)
