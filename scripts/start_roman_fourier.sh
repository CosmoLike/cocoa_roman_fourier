# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if [ -z "${IGNORE_COSMOLIKE_ROMANY1_CODE}" ]; then

  if [ -z "${ROOTDIR}" ]; then
    source start_cocoa.sh || { pfail 'ROOTDIR'; return 1; }
  fi

  # Parenthesis = run in a subshell
  ( source "${ROOTDIR:?}/installation_scripts/flags_check.sh" ) || return 1;

  export LD_LIBRARY_PATH="${ROOTDIR:?}/projects/roman_fourier/interface":${LD_LIBRARY_PATH}

  export PYTHONPATH="${ROOTDIR:?}/projects/roman_fourier/interface":${PYTHONPATH}

  if [ -n "${COSMOLIKE_DEBUG_MODE}" ]; then
      export SPDLOG_LEVEL=debug
  else
      export SPDLOG_LEVEL=info
  fi

fi