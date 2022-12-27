#!/bin/bash

do_line(){
  printf "_%.0s" $(seq $(tput cols))
}

do_set-env(){
  if [[ "$1" =~ "test" ]]
  then
    export DEBUG=1
    export CMAKE_ARGS="-DWITH_COVERAGE=ON -DWITH_DEBUG_INFO=OFF -DMAKE_STUBS=ON"
  fi
  [[ "$1" =~ "dev" ]] && export DEV=1
  export CMAKE_BUILD_PARALLEL_LEVEL=1
}

# do_pip-install(){
#   echo "Executing:" pip install $1
#   printf "[.1] Virtual environment: $VIRTUAL_ENV\n"
#   do_set-env
#   pip install $1 -v
# }

do_manual-install(){  
  printf "[.1] Virtual environment: $VIRTUAL_ENV\n"
  do_set-env $1  
  # Manually ensuring build/test dependencies
  pip install pybind11 pybind11-stubgen || exit 2
  [[ "$1" =~ "test" ]] && \
    pip install pytest pytest-steps pytest-sugar coverage flake8 Pillow numpy \
    || exit 2
#   pip install -e .[$depends] # Should work but fails on various levels
  python setup.py develop # deprecated but functional. Go Python...
}

cmd_pretty-tree(){  # Just a regular tree but without .git folder
  where=$1
  [ -z $1 ] && where=$(pwd)
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
  rm -rf package.{dot,pdf} src/abrain/core/functions/*{eps,png,svg}
  rm -rf tests-results/
  rm -rf _venv*
  rm -f src/abrain/_cpp.*.so
  rm -rf doc/_build doc/src/_autogen doc/src/logo/logo.{pdf,svg}
  find . -name 'abrain.egg-info' | xargs rm -rf
  find src -name "__init__.pyi" | xargs rm -rf
  find src -empty -type d -delete
  rm sample_*
  
  cmd_clean
}
# 
# cmd_install(){ # Regular install (to current virtual env)
#   do_pip-install .
# }
# 
# cmd_install-tests(){  # Install with test in standard location
#   do_pip-install '.[test]'
# }

cmd_install-dev(){  # Editable install (without pip)
  do_manual-install 'dev'
}

cmd_install-dev-tests(){ # Editable install with tests
  do_manual-install 'dev-test'
}

cmd_pytest(){  # Perform the test suite (small scale with evolution)
  out=tests-results
  cout=$out/coverage
  rm -rf $out
  
  lcov --zerocounters --directory .
    
  pycoverage=$cout/py.coverage.info
  coverage run --branch --data-file=$(basename $pycoverage) \
    --source=. --omit "tests/conftest.py,setup.py" \
    -m \
    pytest --durations=10 --basetemp=$out -x -ra $@ || exit 2
  mkdir -p $cout # pytest will have cleared everything. Build it back
  mv $(basename $pycoverage) $pycoverage
  coverage report --data-file=$pycoverage --fail-under=100
  coverage html --data-file=$pycoverage -d $cout/html/python
  
  cppcoverage=$cout/cpp.coverage.info
  lcov --capture --no-external --directory . --output-file $cppcoverage
  lcov --remove $cppcoverage '*_bindings*' -o $cppcoverage
  lcov --list $cppcoverage
  genhtml -q --demangle-cpp -o $cout/html/cpp/ $cppcoverage
}

cmd_test_installs(){ # Attempt at ensuring that everything works fine. WIP (RIP)
  home=$(pwd)
  tmp=/tmp/abrain_test_install
  mkdir -p $tmp
  cd $tmp
  echo "Moved to $tmp"
  
  rm -rf Py-NeuroEvo*
  echo "Cleaned $tmp"
  
  for t in '-dev' '' '-tests' '-dev-tests'
  do
    do_line
    echo "Testing '$t' in $tmp"
    do_line
    echo
    
  #   git clone --recurse-submodules git@github.com:kgd-al/Py-NeuroEvo.git
    rsync -r --exclude={build*/,.*,*results,__pycache__} $home . 
    echo "$home -> ."

    mv Py-NeuroEvo Py-NeuroEvo$t
    cd Py-NeuroEvo$t
    rm -rf _venv
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
    
    cat <<EOF > tester.py
from random import Random
from abrain._cpp.phenotype import CPPN
from abrain.core.genome import Genome

def test_valid_install():
  rng = Random(0)
  g = Genome.random(rng)
  while len(g.nodes) == 0:
    g.mutate(rng)
  g.to_dot("test", "png")
EOF
    pip install pytest
    pytest -x -s tester.py || exit 3
    
    cd ..
    deactivate
    echo
  done 2>&1 | tee install.log
  
  printf "\n\033[32mAll good!\033[m\n"
}

cmd_doc(){  # Generate the documentation
# also requires sphinx and ??
  out=doc/_build
  mkdir -p $out
#   nitpick=-n
  sphinx-build doc/src/ $out/html -b html $nitpick -W $@ 2>&1 \
  | tee $out/log
}

help(){
  echo "Set of commands to help managing/installing/cleaning this repository"
  do_line
  printf "\nAvailable commands:\n"
  sed -n 's/^cmd_\(.*\)(){ *\(#\? *\)\(.*\)/\1|\3/p' $0 | column -s '|' -t
}

# source venv.sh
if [ $1 == "-h" ]
then
  help
  exit 0
else
  cmd="cmd_$1"
  echo "Making" $cmd
  shift
  eval $cmd $@
fi
