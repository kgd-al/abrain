#!/bin/bash

do_line(){
  printf "_%.0s" $(seq $(tput cols))
}

do_set-env(){
  CMAKE_ARGS="-DMAKE_STUBS=ON "
  if [[ "$1" =~ "tests" ]]
  then
    export DEBUG=1
    CMAKE_ARGS="$CMAKE_ARGS -DWITH_COVERAGE=ON -DWITH_DEBUG_INFO=OFF"
  else
    CMAKE_ARGS="$CMAKE_ARGS -DWITH_COVERAGE=OFF -DWITH_DEBUG_INFO=OFF"
  fi
  export CMAKE_ARGS
  [[ "$1" =~ "dev" ]] && export DEV=1
  export CMAKE_BUILD_PARALLEL_LEVEL=4
}

do_pip-install(){
  echo "Executing:" pip install $1
  printf "[.1] Virtual environment: $VIRTUAL_ENV\n"
  do_set-env "$1"
  
#   export VERBOSE=1
  verbose="-v" #vv"
  
  pip install $1 $verbose
}

do_manual-install(){  
  printf "[.1] Virtual environment: $VIRTUAL_ENV\n"
  do_set-env "$1"
  # Manually ensuring build/test dependencies
  pip install pybind11[global] pybind11-stubgen || exit 2
  if [[ "$1" =~ "tests" ]]
  then
    pip install pytest pytest-steps pytest-sugar coverage flake8 Pillow numpy \
    || exit 2
  fi
#   pip install -e .[$depends] # Should work but fails on various levels

  python setup.py develop # deprecated but functional. Go Python...

#  # Still not working as well as develop
#  pip wheel -e . --no-build-isolation --no-deps \
#  && pip install --no-deps --force-reinstall abrain*.whl
}

cmd_pretty-tree(){  # Just a regular tree but without .git folder
  where=${1:-}
  [ -z $where ] && where=$(pwd)
  tree -a -I '.git*' $where
}

cmd_clean(){  # Remove most build artifacts
  rm -rf .pytest_cache .tox .idea
  find . -type d -a \
    \( -name build -o -name '__pycache__' -o -name "*egg-info" \) \
    | xargs rm -rf

  echo "Cleaned tree:"
  cmd_pretty-tree
}

cmd_very-clean(){  # Remove all artifacts. Reset to a clean repository
  echo "very clean"
  rm -rf src/abrain/_cpp/misc/constants.h
  rm -rf package.{dot,pdf}
  rm -rf tests-results/
  rm -rf _venv*
  rm -f src/abrain/_cpp.*.so
  rm -rf doc/_build doc/src/_autogen/errors.rst doc/src/logo/logo.{pdf,svg}
  rm -rf *build*/
  rm -rf tmp
  rm -rf *.whl
  find . -name 'abrain.egg-info' | xargs rm -rf
  find src -name "*.pyi" | xargs rm -rf
  find src -empty -type d -delete
  rm -f sample_*
  
  cmd_clean
}

cmd_install-user(){ # Regular install (to current virtual env)
  do_pip-install .
}

cmd_install-docs(){ # Documentation install (for read the docs)
  do_pip-install '.[docs]'
}

cmd_install-tests(){  # Install with test in standard location
  do_pip-install '.[tests]'
}

cmd_install-dev(){  # Editable install (with pip)
  do_pip-install '-e .[docs,tests]'
#   do_manual-install 'dev-test-doc'
}

cmd_install-cached(){ # Editable install (without pip and cached build folder)
  type=${1:-'dev-tests-doc'}
  echo "Building for type '$type'"
  do_manual-install "$type"
}

cmd_clion-setup(){ # Print out configuration options for CLion
  log=.log
  config=clion_config
  export CLION_CONFIG=$config
  cmd_install-cached "" >$log 2>&1
  ok=$?
  if [ $ok -eq 0 ]
  then
    printf "\033[32mLocal build successful.\033[0m\n"
    rm -f $log
  else
    printf "\033[31mLocal build Failed.\033[0m\n"
    echo "Details in $log"
  fi

  sed -i 's|</project>|  <component name="CMakeWorkspace" PROJECT_DIR="$PROJECT_DIR$" />\n&|' .idea/misc.xml

  echo "CLion setup instructions (from $config)"
  echo "-----------------"
  cat $config
  echo "-----------------"
}

cmd_pytest(){  # Perform the test suite (small scale with evolution)
  out=tmp/tests-results
  cout=$out/coverage
  rm -rf $out
  mkdir -p $out
  
  lcov --zerocounters --directory .

  pycoverage=$cout/py.coverage.info
  coverage run --branch --data-file=$(basename $pycoverage) \
    --source=. --omit "setup.py,tests/conftest.py,examples/*.py,tests/test_examples.py" \
    -m \
    pytest --durations=10 --basetemp=$out -x -ra "$@" || exit 2
  mkdir -p $cout # pytest will have cleared everything. Build it back
  mv $(basename $pycoverage) $pycoverage
  coverage report --data-file=$pycoverage
  coverage html  --fail-under=100 --data-file=$pycoverage -d $cout/html/python
  
  cppcoverage=$cout/cpp.coverage.info
  lcov --capture --no-external --directory . --rc lcov_branch_coverage=1 --output-file $cppcoverage
  lcov --remove $cppcoverage '*cppn.h' -o $cppcoverage
  lcov --remove $cppcoverage '*_bindings*' -o $cppcoverage
  lcov --list $cppcoverage
  genhtml -q --demangle-cpp --branch-coverage -o $cout/html/cpp/ $cppcoverage
}

cmd_test_installs(){ # Attempt at ensuring that everything works fine. WIP
  home=$(pwd)
  tmp=/tmp/abrain_test_install
  mkdir -p $tmp
  cd $tmp
  echo "Moved to $tmp"

  rm -rf abrain* _venv*
  echo "Cleaned $tmp"
  
  # OK: '-user' '-docs' '-tests' '-dev'
  for t in '-user' '-docs' '-tests' '-dev'
  do
    do_line
    echo "Testing '$t' in $tmp"
    do_line
    echo
    
    dir=abrain$t
  #   git clone --recurse-submodules git@github.com:kgd-al/abrain.git
    mkdir -pv $dir
    rsync -r --exclude={build*/,.*,*results,__pycache__} $home/* $dir/ 
    echo "$home -> $dir"

    cd $dir
    $home/$0 very-clean
    
    printf "[0] Virtual environment: $VIRTUAL_ENV\n"
    python3 -m venv ../_venv$t || exit 1
    printf "[1] Virtual environment: $VIRTUAL_ENV\n"
    source ../_venv$t/bin/activate || exit 1
    printf "[2] Virtual environment: $VIRTUAL_ENV\n"
    python3 -m ensurepip --upgrade || exit 1
    printf "[3] Virtual environment: $VIRTUAL_ENV\n"
    pip --version
    python3 -m pip install --upgrade pip || exit 1
    printf "[4] Virtual environment: $VIRTUAL_ENV\n"

    time eval "cmd_install$t" || exit 2
    printf "[5] Virtual environment: $VIRTUAL_ENV\n"
    
    python examples/basics.py || exit 3
    
    [[ "$t" =~ "docs" ]] && { cmd_doc || exit 3; }
    [[ "$t" =~ "test" ]] && { cmd_pytest --small-scale --test-evolution || exit 3; }
    
    pip list
    
    cd ..
    deactivate
    echo
    
  done 2>&1 | tee install.log
  
  r=${PIPESTATUS[0]}
  if [ "$r" -eq 0 ]
  then
    printf "\n\033[32mAll good!\033[0m\n"
  else
    printf "\n\033[31;5m/!\\ Install failed /!\\ \033[0m\n"
    printf "> With exit code \033[31m$r\033[0m\n"
    exit $r
  fi
}

do_doc_prepare() {
  out=doc/_build
  mkdir -p $out
}

cmd_doc(){  # Generate the documentation
# also requires sphinx and sphinx-pyproject
  do_doc_prepare
  nitpick=-n
  sphinx-build doc/src/ $out/html -b html $nitpick -W $@ 2>&1 \
  | tee $out/log
}

cmd_autodoc(){  # Generate the documentation continuously
  do_doc_prepare
  sphinx-autobuild doc/src/ $out/html -aE --watch src --watch examples \
    --color -W --keep-going -w $out/errors --ignore '*_auto*' \
    --pre-build 'bash -c "clear; date;"' $@
}

cmd_before-deploy(){  # Run a lot of tests to ensure that the package is clean
  set -euo pipefail
  shopt -s inherit_errexit
  ok=1
  check(){
    if [ $ok -ne 0 ]
    then
      printf "\033[31mPackage is not ready to deploy."
      printf " See error(s) above.\033[0m\n"
    else
      printf "\033[32mPackage checks out.\033[0m\n"
    fi
  }
  trap check exit
  test_dir=$(realpath "../__abrain_deploy_test__/")
  cmd_very-clean
  rm -rf "$test_dir"
  mkdir -p "$test_dir"
  cp -r . "$test_dir/abrain"
  cd "$test_dir/abrain"
  virtualenv "../venv"
  source "../venv/bin/activate"
  echo "$VIRTUAL_ENV"
  pip install --upgrade pip
  TEST=1 pip install ".[all-tests, docs]" -v
  cmd_pytest --small-scale --test-evolution --test-examples
  flake8 src tests
  cmd_doc
  rm -rf "$test_dir"
  ok=0
}

help(){
  echo "Set of commands to help managing/installing/cleaning this repository"
  do_line
  printf "\nAvailable commands:\n"
  sed -n 's/^cmd_\(.*\)(){ *\(#\? *\)\(.*\)/\1|\3/p' $0 | column -s '|' -t
}

if [ $# -eq 0 ]
then
  echo "No commands/arguments provided"
  help
  exit 1
fi

if [ "$1" == "--venv" ]
then
  source $2/bin/activate
  shift 2
fi

if [ -z $VIRTUAL_ENV ]
then
  echo "Refusing to work outside of a virtual environment."
  echo "Activate it beforehand or provide it with the --venv <path/to/venv> option."
  exit 1
fi

if [ "$1" == "-h" ]
then
  help
  exit 0
else
  cmd="cmd_$1"
  echo "Making" $cmd
  shift
  eval $cmd "$@"
fi
