#!/bin/sh

echo "reconf 2"
./run_reconf_2.sh $1
sleep 20


echo "reconf 5"
./run_reconf_5.sh $1
sleep 20


echo "reconf 10"
./run_reconf_10.sh $1
