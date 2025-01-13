export PYTHONPATH="${PYTHONPATH}:{PWD?}"
export ENV_SHOW_PLOT="False"
for f in scripts/validating_scripts/*.py; do ../LinuxBASH/bin/python "$f"; done
../LinuxBASH/bin/python "utils/plot_validation.py"