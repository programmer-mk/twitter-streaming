#!/bin/bash

set -e  # exit immediately if a command exits with a non-zero status
set -x  # print all executed commands on terminal

PROJECT_DIR=$1
MODULE_NAME=$2
VERSION=$3

echo $PROJECT_DIR
echo $MODULE_NAME
echo $VERSION

#$PROJECT_DIR/wait-for-it.sh zookeeper.twitter.streaming.data.com:2181 -t 30 -- echo "Zookeeper started"
#$PROJECT_DIR/wait-for-it.sh kafka.twitter.streaming.data.com:9092 -t 30 -- echo "Kafka started"

java -jar $PROJECT_DIR/"$MODULE_NAME"-assembly-"$VERSION".jar;