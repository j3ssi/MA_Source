#!/bin/sh

echo "reconf 2 1"
./run_reconf_2.sh 1
sleep 20


echo "reconf 5 1"
./run_reconf_5.sh 1
sleep 20


echo "reconf 10 1"
./run_reconf_10.sh 1


echo "reconf 2 2"
./run_reconf_2.sh 2
sleep 20


echo "reconf 5 2"
./run_reconf_5.sh 2
sleep 20


echo "reconf 10 2"
./run_reconf_10.sh 2


echo "reconf 2 3"
./run_reconf_2.sh 3
sleep 20


echo "reconf 5 3"
./run_reconf_5.sh 3
sleep 20


echo "reconf 10 3"
./run_reconf_10.sh 3



echo "reconf 2 4"
./run_reconf_2.sh 4
sleep 20


echo "reconf 5 4"
./run_reconf_5.sh 4
sleep 20


echo "reconf 10 4"
./run_reconf_10.sh 4


echo "reconf 2 5"
./run_reconf_2.sh 5
sleep 20


echo "reconf 5 5"
./run_reconf_5.sh 5
sleep 20


echo "reconf 10 5"
./run_reconf_10.sh 5

