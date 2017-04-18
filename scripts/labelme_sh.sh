#!/bin/bash

input=$1
output=$2

# source deactivate

labelme $input -O $output

# source activate env2
