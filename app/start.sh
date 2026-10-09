#!/bin/sh

python init_db.py

exec gunicorn --workers 4 --bind 0.0.0.0:5000 server:app