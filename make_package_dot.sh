#!/bin/bash

out=package.dot
echo "digraph PyNE {" > $out
echo -e "\trankdir=\"BT\";" >> $out
echo -e "\tfontname=\"Courier\";" >> $out
echo -e "\tfontsize=\"18\";" >> $out

tmp=.$(basename $0)

echo > $tmp.links
find . -type f | grep -vP "(extern|build)" | grep -P ".*\.(cpp|h|hpp)" | sort \
| while read file
do
  file=$(sed 's|^\./||' <<< $file)
  for include in $(grep "#include[ ]*\"" $file | sed 's|.*"\(.*\)"|\1|')
  do
    grep -q "pybind11" <<< $include && continue 
    path=$(dirname $file)/$include
    realpath=$(realpath --relative-to=. -e $path)
    if [ -z "$realpath" ]
    then
      echo "Invalid include '$include' in '$file'"
      exit 2
    fi
    echo -e "\t$realpath -> $file;" | tr './' '_' >> $tmp.links
  done

  echo $file >> $tmp.nodes
done || exit 2

# Collect folders to get potential modules
modules=(pyne_cpp \
  $(cd python && find . -name "*.py" | xargs -I{} dirname {} | sed 's|^./||' | sort | uniq))  
echo >> $tmp.links
find python -name "*.py" \
| while read file
do
#   echo $file
  grep import $file | while read line
  do
    module=$(cut -d ' ' -f 2 <<< "$line")    
    firstmodule=$(cut -d '.' -f 1 <<< "$module")
    [[ ! " ${modules[*]} " =~ " $firstmodule " ]] && continue
#     echo -e "\t> $module"
    
    refs=()
    
    # c++ main module
    if [ $module == "pyne_cpp" ]
    then
      refs=(pybind/module.cpp)
      
    # c++ binding -> reference c++ file
    elif [[ $firstmodule =~ "pyne_cpp" ]]
    then
      for obj in $(sed -e 's/.* import //' -e 's/ as .*//' <<< "$line" | tr ',' ' ')
      do
#         echo "Looking for $obj"
        ref=$(grep -r "struct $obj\b" $(sed 's/pyne_//' <<< "$module" | tr . /)* \
          | cut -d ':' -f 1)
        refs+=($ref)
      done
      
    # Must an honest python import
    else
      refs=("python/"$(tr '.' '/' <<< $module).py)
    fi

    for ref in ${refs[@]}
    do
#       echo -e "\t\033[32m$file -> $refs\033[0m"
      echo -e "\t$ref -> $file;" | tr '/.' '_' >> $tmp.links
    done
  done
  
  echo $file >> $tmp.nodes
done

python3 - >> $out <<EOF
import pathlib

def process (keys, array, l=""):
#   print(f"{l}Processing {keys}")
  
  k = keys[0]
  keys.pop(0)
  if k not in array:
    array[k] = {}
  if (len(keys) > 0):
    process(keys, array[k], l+"\t")
  else:
    array[k] = 1;
  
def output(array, path="", l="\t"):
  for k in array:
    if hasattr(array[k], "__len__"):
      print(f"{l}subgraph cluster_{k}"+ " {")
      print(f"{l}\tlabel=\"{k}\";")
      output(array[k], path+k+"_", l+"\t")
      print(l + "}")
    else:
      dot_name = path + k.replace(".", "_")
      ext = pathlib.Path(k).suffix[1:]
      if ext == "cpp":
        color = "#000000"
      elif ext == "h":
        color =  "#00F000"
      elif ext == "hpp": 
        color = "#007000"
      else:
        color = "#0000F0"
      print(f"{l}{dot_name} [label=\"{k}\", style=filled, fillcolor=\"{color}30\" color=\"{color}\"];") 
  
data = {}
with open('$tmp.nodes', 'r') as file:
  for line in file:
    line = line.rstrip()
  #   print(line)
    process(line.split("/"), data)
  output(data)
  
EOF

cat $tmp.links >> $out

echo "}" >> $out

# nl -w 2 -b a $out | sed 's/\t/  /g' | cut -c -$(tput cols)

pdf=$(basename $out .dot).pdf
dot -Tpdf $out -o $pdf
echo "Generated $pdf"

rm -f $tmp*
