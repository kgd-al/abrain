#!/bin/bash

if test -z "${VIRTUAL_ENV}"
then
  source ${HOME}/.python/venvs/default/bin/activate
  echo "Activated virtual environment '${VIRTUAL_ENV}'"
fi
