# Big_data_project23

Spark version==3.4.1

# Comandi utili da terminale

## start zookeeper

~/kafka_2.13-3.4.0 $ bin/zookeeper-server-start.sh config/zookeeper.properties

## start kafka server

~/kafka_2.13-3.4.0 $ bin/kafka-server-start.sh config/server.properties

## create topic 

~/kafka_2.13-3.4.0 $ bin/kafka-topics.sh --create --bootstrap-server {your_ip_address}:9092 --replication-factor 1 --partitions 1 --topic TestTopic

## start producer kafka

kafka-console-producer.sh --bootstrap-server localhost:9092 --topic test_topic --property "parse.key=true" --property "key.separator=:"

## run spark consumer

spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:VERSIONE_SPARK         PATH_DELLO_SCRIPT

## start hadoop

start-dfs.sh && start-yarn.sh


