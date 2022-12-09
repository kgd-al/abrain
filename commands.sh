#!/bin/bash

do_line(){
  printf "_%.0s" $(seq $(tput cols))
}

do_install(){
  pip install $1
}

cmd_install(){ # Regular install (to current virtual env)
  do_install .
}

cmd_install-dev(){  # Editable (strict) install
  do_install '-e . --config-settings editable_mode=strict'
}

cmd_install-tests(){  # Install with test in standard location
  do_install '.[tests]'
}

cmd_install-dev-tests(){ # Editable install with tests
  do_install '-e .[tests] --config-settings editable_mode=strict'  
}

cmd_pytest(){
  coverage run --branch --source=. -m \
    pytest --durations=10 --basetemp=tests-results -x -rs $@
  coverage report
  coverage html
}

cmd_test_installs(){
  tmp=$(mktemp -d --tmpdir "pyne_test_install_XXXX")
  cd $tmp
  echo "Moved to $tmp"
  
  git clone git@github.com:kgd-al/Py-NeuroEvo.git
  cd Py-NeuroEvo
  
  for t in '' '-dev' '-tests' '-dev-tests'
  do
    do_line
    echo "Testing '$t' in $tmp"
    do_line
    echo
    
    echo "Foo on stderr" >&2
    echo "Foo on stdout"
    
    virtualenv _venv$t && source _venv$t/bin/activate || exit 1
    pip install --upgrade pip

    time eval "cmd_install$t" || exit 2
    
    cat <<EOF > tester.py
      from random import Random
      from pyne._cpp import CPPN
      from pyne.core.genome import Genome
      
      def test_valid_install():
        rng = Random(0)
        g = Genome()
        while len(g.nodes) == 0:
          g.mutate(rng)
        g.to_dot("test", "png")
EOF
    pip install pytest
    pytest -x -s
    
    deactivate
    echo
  done | tee install.log
}

help(){
  echo "Set of commands to help managing/installing/cleaning this repository"
  do_line
  printf "\nAvailable commands:\n"
  sed -n 's/^cmd_\(.*\)(){ *\(#\? *\)\(.*\)/\1|\3/p' $0 | column -s '|' -t
}

echo $1
if [ $1 == "-h" ]
then
  help
  exit 0
else
  eval "cmd_$1" $@
fi
done
