package streaming

import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.streaming.dstream.InputDStream
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010.{CanCommitOffsets, ConsumerStrategies, HasOffsetRanges, KafkaUtils, LocationStrategies}
import org.slf4j.{Logger, LoggerFactory}


object TweetKafkaStreaming {
  // Kafka's broker address
  val brokers: String = "kafka:9092"
  // Each stream uses a separate group.id
  val groupId: String = "kafka_spark_straming"
  // The topic to be consumed, accepts an array, and can pass in multiple topics
  val topics: Array[String] = Array("tweets")
  // Used to record logs
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  val kafkaParams: Map[String, Object] = Map[String, Object](
    "bootstrap.servers" -> brokers,
    "key.deserializer" -> classOf[StringDeserializer],
    "value.deserializer" -> classOf[StringDeserializer],
    "group.id" -> groupId,
    "auto.offset.reset" -> "latest",
    "enable.auto.commit" -> (false: java.lang.Boolean)
  )

  def getKafkaStream(ssc: StreamingContext, topics: Array[String]): InputDStream[ConsumerRecord[String, String]] ={
    // Use KafkaUtils.createDirectStream to create a data stream
    val stream: InputDStream[ConsumerRecord[String, String]] = KafkaUtils.createDirectStream[String, String](
      ssc,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](topics, kafkaParams)
    )
    stream
  }

  def main(args: Array[String]): Unit = {

    val sparkConf: SparkConf = new SparkConf().setMaster("local[*]").setAppName(this.getClass.getName)
    val ssm: StreamingContext = new StreamingContext(sparkConf, Seconds(10))

    val inputDStream: InputDStream[ConsumerRecord[String, String]] = getKafkaStream(ssm, topics)

    inputDStream.foreachRDD { rdd =>
      // Get the offset
      val offsetRanges = rdd.asInstanceOf[HasOffsetRanges].offsetRanges
      rdd.foreachPartition {iter =>
        iter.foreach { consumerRecord =>
          val key: String = consumerRecord.key()
          val value: String = consumerRecord.value()
          println(s"key: ${key}, value: ${value}")
        }
      }
      // After a period of time, after the calculation is completed, the offset is submitted asynchronously
      inputDStream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
    }

    ssm.start()
    ssm.awaitTermination()
  }
}