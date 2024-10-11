export PYTHONPATH="${PYTHONPATH}:{PWD?}"
export ENV_SHOW_PLOT="False"
for f in training_scripts/*.py; do ../LinuxBASH/bin/python "$f"; done