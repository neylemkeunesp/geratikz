#!/bin/bash
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/lemke/geratikz
cd /home/lemke/geratikz
exec python3 src/main.py
