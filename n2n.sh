#!/bin/sh


echo "Net2Net 1"
./net2net1.sh 1
sleep 20

echo "Net2Net Random 1"
./net2ner1Rnd.sh 1
sleep 20

echo "Net2Net 2"
./net2net1.sh 2
sleep 20

echo "Net2Net Random 2"
./net2ner1Rnd.sh 2
sleep 20

echo "Net2Net 3"
./net2net1.sh 3
sleep 20

echo "Net2Net Random 3"
./net2ner1Rnd.sh 3
sleep 20

echo "Net2Net 4"
./net2net1.sh 4
sleep 20

echo "Net2Net Random 4"
./net2ner1Rnd.sh 4
sleep 20

echo "Net2Net 5"
./net2net1.sh 5
sleep 20

echo "Net2Net Random 5"
./net2ner1Rnd.sh 5
sleep 20
