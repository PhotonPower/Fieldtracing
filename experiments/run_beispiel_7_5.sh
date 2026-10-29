#!/usr/bin/env bash
# Full intermediate plane z = f/2 of the example of section 7.5 (#41) on a machine with enough
# memory (≥ 8 GB; the full system a = 12.5 mm needs about 5-6 GB). Written for AlmaLinux 10 (dnf),
# tested on Ubuntu 24.04 with SKIP_DNF=1; any Linux with Python ≥ 3.10. CPU only, no GPU.
#
#   curl -fsSLO https://raw.githubusercontent.com/PhotonPower/Fieldtracing/main/experiments/run_beispiel_7_5.sh
#   bash run_beispiel_7_5.sh                 # installs into ~/fieldtracing-run, runs, writes a log
#
# Environment variables (optional):
#   WORKDIR    working directory              (default ~/fieldtracing-run)
#   REF        Fieldtracing branch or commit  (default main)
#   RADII      stop radii in mm               (default "1.25 2.5 5 12.5")
#   RUN_TESTS  1: also run pytest and pytest -m slow (about 6 min)  (default 0)
#   SKIP_DNF   1: do not install system packages with dnf          (default 0)
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/fieldtracing-run}"
REF="${REF:-main}"
RADII="${RADII:-1.25 2.5 5 12.5}"
RUN_TESTS="${RUN_TESTS:-0}"
SKIP_DNF="${SKIP_DNF:-0}"
REPO="https://github.com/PhotonPower/Fieldtracing.git"

mkdir -p "$WORKDIR"
LOG="$WORKDIR/beispiel_7_5_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "== Fieldtracing: example 7.5, intermediate plane =="
date -Is

# 1. System packages (git, Python with venv, GNU time for the peak memory)
if [ "$SKIP_DNF" != "1" ]; then
  missing=()
  command -v git >/dev/null || missing+=(git)
  command -v python3 >/dev/null || missing+=(python3)
  python3 -m pip --version >/dev/null 2>&1 || missing+=(python3-pip)
  [ -x /usr/bin/time ] || missing+=(time)
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "installing: ${missing[*]}"
    sudo dnf install -y "${missing[@]}"
  fi
fi

# 2. Sources
if [ -d "$WORKDIR/Fieldtracing/.git" ]; then
  git -C "$WORKDIR/Fieldtracing" fetch -q origin
else
  git clone -q "$REPO" "$WORKDIR/Fieldtracing"
fi
cd "$WORKDIR/Fieldtracing"
git checkout -q "$REF"
git pull -q --ff-only 2>/dev/null || true
echo "Fieldtracing commit: $(git rev-parse HEAD)"

# 3. Virtual environment with the pinned versions (numpy, scipy as in the CI)
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-dev.txt
python -m pip install -q -e .

# 4. Machine
echo "--- machine ---"
python --version
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
uname -r
grep -m1 "model name" /proc/cpuinfo || true
nproc
free -g | head -2

# 5. Runs: one process per radius, so that the peak memory belongs to that radius
echo "--- runs ---"
for a in $RADII; do
  echo ">>> a = $a mm"
  /usr/bin/time -v python experiments/beispiel_7_5_zwischenebene.py "$a" 2> "time_$a.txt"
  grep -E "Maximum resident set size|Elapsed \(wall clock\)" "time_$a.txt" | sed 's/^\s*/    /'
done

# 6. Optional: the whole test suite including the expensive tests
if [ "$RUN_TESTS" = "1" ]; then
  echo "--- pytest ---"
  python -m pytest -q -p no:cacheprovider | tail -3
  python -m pytest -q -p no:cacheprovider -m slow --durations=10 | tail -15
fi

echo "--- done ---"
echo "log: $LOG"
