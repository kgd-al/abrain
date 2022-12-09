#!/bin/bash

cd $(dirname $0)
mkdir -pv tmp/
cd tmp

functions="
abs;abs(x)
gaus;e^(-6.25*x*x)
id;x
bsgm;2/(1+e^(-4.9*x))-1
ssgm;1/(1+e^(-4.9*x))
sin;sin(deg(2*x*pi))
step;x<=0?0:1"

for line in $functions
do
    IFS=';' read name func <<< "$line"
    printf "%4s: %s\n" "$name" "$func"
    sed 's|\\def\\func{.*}|\\def\\func{'"$func"'}|' ../template.tex > $name.tex
    latex $name.tex >/dev/null && dvips -F* -s* -q $name.dvi -o $name.eps && eps2eps -f $name.eps ../$name.eps
    convert -density 100 ../$name.eps ../$name.png
#     mv $name.eps ../
done

cd ..
rm -r tmp
ls -lh *ps *.png
          
