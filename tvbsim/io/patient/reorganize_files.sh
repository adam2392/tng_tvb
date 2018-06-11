#!/bin/bash

dirs=(/./*/)

for dir in */; do 
	mkdir -- "$dir/tmp1"; 
done
