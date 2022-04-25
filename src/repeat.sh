#!/bin/bash

for i in $(seq 1 $1)
do
    python3 main.py info.json >> output.log
done
