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
  ' $cppn \
  | sed -e 's/std::fabs/abs/' \
        -e 's/KGD_EXP/exp/' \
        -e 's/fd_//' \
        -e 's/\([0-9.]\)f/\1/g' \
        -e 's/\([0-9]\)\.\([^0-9]\)/\1\2/g' \
    > $listing
  
while read line
do
    IFS=';' read name func desc <<< "$line"
    printf "%4s: %s\n" "$name" "$func"
    
    eps_func=$func
    [ "$func" == "ssgn(x)" ] && eps_func="x < -.25 ? exp(-4*(x+.25)*(x+.25)) - 1 : (x > .25 ? 1 - exp(-4*(x-.25)*(x-.25)) : 0)"
    
    sed 's|\\def\\func{.*}|\\def\\func{'"$eps_func"'}|' ../template.tex > $name.tex
    [ ! -f "../$name.svg" ] \
    && latex $name.tex > log \
    && dvips -F* -s* -q $name.dvi -o $name.eps \
    && eps2eps -f $name.eps ../$name.eps \
    && convert -density 100 ../$name.eps ../$name.png \
    && convert ../$name.eps ../$name.svg
    
    latex_func=$(echo $func | sed \
      -e 's|exp(\([^)]*\))|e^{\1}|' \
      -e 's|\(.*\)<=\(.*\)?\(.*\):\(.*\)|\\begin{cases}\3 \& \\text{if }\1 \\leq \2\\\\ \4 \& \\text{otherwise}\\end{cases}|' \
      -e 's|x\*x|x^2|' \
      -e 's/abs(\(.*\))/|\1|/' \
      -e 's|*||' \
      -e 's|ssgn(x)|\\begin{cases} e^{-(x+1)^2}-1 \& \\text{if } x \\lt -1 \\\\1 - e^{-(x-1)^2} \& \\text{if } x \\gt 1 \\\\ 0 \& \\text{otherwise} \\end{cases}|')
    python_func=$(echo $func | sed \
      -e 's|sin|np.sin|' \
      -e 's|exp|np.exp|' \
      -e 's|x <= .*?.*:.*|np.heaviside(x, 0)|' \
      -e 's|x\*x|x**2|' \
      -e 's|ssgn(x)|np.piecewise(x, [x < -1, 1 < x], [lambda x: np.exp(-(x+1)**2) -1, lambda x: 1 - np.exp(-(x-1)**2), 0])|')
      
    printf "$name:\t$desc\n\t  [base] $func\n\t  [math] $latex_func\n\t[python] $python_func\n\n"
    
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
           "       plt.plot(x, $python_func)" \
           "" >> $doc 
done < $listing

cd ..
rm -r tmp
ls -lh *ps *.png
          
