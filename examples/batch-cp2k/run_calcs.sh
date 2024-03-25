#! /bin/bash

for i in $(find ./production/ -mindepth 1 -type d); do
    cd $i
    cp2k -i in.cp2k
    cd -
done

