#!/bin/bash

export LC_ALL=C

cppn=$(realpath src/abrain/_cpp/phenotype/cppn.cpp)
echo $cppn

doc=doc/src/_autogen/functions.rst
mkdir -p $(dirname $doc)
doc=$(realpath $doc)
version=$(grep "^version" pyproject.toml | cut -d "=" -f 2 | tr -d '" ')
printf ".. note:: Automatically extracted from sources on %s for version %s\n\n" \
  "$(date)" "$version" > $doc
echo $doc

cd $(dirname $0)
mkdir -pv tmp/
cd tmp

listing=functions

awk '
    /CPPN::functions {/{ inmap=1 }
    inmap && /^}/{ inmap = 0 }
    !/F\(/{next;}
    inmap {
#       print $0
      match($0, /.*F\( *"(.*)", (.*)\),.* \/\/ (.*)/, tokens)
      printf("%s;%s;%s\n", tokens[1], tokens[2], tokens[3])
    }
  ' $cppn > $listing
  
declare -A data=(
  ["L","id"]="x" ["B","id"]=".5*x"
  ["L","abs"]="|x|" ["B","abs"]="abs(x)"
  ["L","sin"]="sin(2x)" ["B","sin"]="sin(100*x)"
  
  ["L","step"]="0 &\ \text{if } x \leq 0\\\\1 &\ \text{otherwise}"
  ["B","step"]="x <= 0 ? 0 : 1"
  
  ["L","gaus"]="e^{-6.25x^2}" ["B","gaus"]="exp(-6.25*x*x)"
  
  ["L","ssgm"]="\frac{1}{1+e^{-4.9x}}"
  ["B","ssgm"]="1/(1+exp(-4.9*x))"
  
  ["L","bsgm"]="\frac{2}{1+e^{-4.9x}} - 1"
  ["B","bsgm"]="2/(1+exp(-4.9*x)) - 1"
  
  ["L","ssgn"]="e^{-(x+1)^2} - 1 &\ \text{if } x \lt -1 \\\\1 - e^{-(x-1)^2} &\ \text{if } x \gt  1 \\\\0 &\ \text{otherwise}"
  ["B","ssgn"]="x < -.25 ? exp(-4*(x+.25)*(x+.25)) - 1 : (x > .25 ? 1 - exp(-4*(x-.25)*(x-.25)) : 0)"  
)
  
while read line
do
    IFS=';' read name func desc <<< "$line"
    
    if [ ${data["B","$name"]+_} ]
    then
      printf "%4s: %s\n" "$name" "$func"
    else
      echo "No data found for $name"
      echo ".. warning:: Undocumented function $name" >> $doc
      continue
    fi
    
    badge_func=${data["B","$name"]}
    latex_func=${data["L","$name"]}

    sed 's|\\def\\func{.*}|\\def\\func{'"$badge_func"'}|' ../template.tex > $name.tex
    pdflatex -shell-escape $name.tex > log \
    && convert -density 300 $name.svg ../$name.png
    mv $name.svg ../
    
    printf "%4s: %s\n" "$name" "$desc"
    printf "%10s %s\n" "[base]" "$func"
    printf "%10s %s\n" "[badge]" "$badge_func"
    printf "%10s %s\n" "[latex]" "$latex_func"
    
    header=$(printf "_%.0s" $(seq 1 ${#desc}))
    printf "%s\n" \
           "$desc" \
           "$header" \
           "" \
           ".. grid:: auto" \
           "  :gutter: 0" \
           "" \
           "  .. grid-item-card::" \
           "    :columns: 2" \
           "    :text-align: right" \
           "" \
           "    .. image:: $(realpath --relative-to=$(dirname $doc) ../$name.svg)" \
           "" \
           "  .. grid-item-card::" \
           "    :columns: 10" \
           "" \
           "    .. math:: $latex_func" \
           "" \
           "  .. grid-item-card::" \
           "    :columns: 2" \
           "    :text-align: right" \
           "" \
           "    $name" \
           "" \
           "  .. grid-item-card::" \
           "     :columns: 10" \
           "" \
           "     .. plot::" \
           "       :height: 10em" \
           "" \
           "       plt.plot(x, [functions['$name'](x_) for x_ in x])" \
           "" >> $doc 
done < $listing

cd ..
rm -r tmp
ls -lh *.png *.svg
          
