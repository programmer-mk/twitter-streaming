import java.time.Instant

import com.fasterxml.jackson.annotation.JsonProperty
import com.vader.sentiment.analyzer.SentimentAnalyzer
import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.spark.streaming.dstream.InputDStream
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.{Seconds, StreamingContext}
import streaming.Util

case class Tweet(id: Long, t_key: String, text: String, processed_text: String, created: String)

object TweetKafkaStreaming {
  // Kafka's broker address
  val brokers: String = "kafka:9092"
  // Each stream uses a separate group.id
  val groupId: String = "kafka_spark_streaming"
  // The topic to be consumed, accepts an array, and can pass in multiple topics
  val topics: Array[String] = Array("tweet-upload-5")
  // Used to record logs
  val log: Logger = Logger.getLogger(getClass.getName)

  val kafkaParams: Map[String, Object] = Map[String, Object](
    "bootstrap.servers" -> brokers,
    "key.deserializer" -> classOf[StringDeserializer],
    "value.deserializer" -> classOf[TweetDeserializer],
    "group.id" -> groupId,
    "auto.offset.reset" -> "latest",
    "enable.auto.commit" -> (false: java.lang.Boolean)
  )

  def getKafkaStream(ssc: StreamingContext, topics: Array[String]): InputDStream[ConsumerRecord[String, UserTweet]] ={
    // Use KafkaUtils.createDirectStream to create a data stream
    val stream: InputDStream[ConsumerRecord[String, UserTweet]] = KafkaUtils.createDirectStream[String, UserTweet](
      ssc,
      LocationStrategies.PreferConsistent,
      ConsumerStrategies.Subscribe[String, UserTweet](topics, kafkaParams)
    )
    stream
  }

  def computePolarity(input: String): Float = {
    val sentimentAnalyzer = new SentimentAnalyzer(input)
    sentimentAnalyzer.analyze()
    val polarities = sentimentAnalyzer.getPolarity
    val compoundPolarity = polarities.get("compound")
    compoundPolarity
  }

  def main(args: Array[String]): Unit = {
    val sparkConf: SparkConf = new SparkConf().setMaster("spark://spark-master:7077").setAppName(getClass.getName)
    val ssm: StreamingContext = new StreamingContext(sparkConf, Seconds(30))
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", System.getenv("AWS_ACCESS_KEY_ID"))
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", System.getenv("AWS_SECRET_ACCESS_KEY"))
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    val inputDStream: InputDStream[ConsumerRecord[String, UserTweet]] = getKafkaStream(ssm, topics)
    inputDStream.foreachRDD { rdd =>
      // Get the offset
      val offsetRanges = rdd.asInstanceOf[HasOffsetRanges].offsetRanges
      if(rdd.count() > 0) {
        log.info(s"rdd size: ${rdd.count()}")
        val rdds: RDD[Seq[String]] = rdd.map(x => Seq(x.key(), s"${computePolarity(Util.cleanDocument(x.value().getText))}",
          Util.cleanDocument(x.value().getText), x.value().getCreated.toString)).cache()
        rdds.map { seq =>
          seq.mkString(",")
        }.repartition(1).saveAsTextFile(s"s3a://test-spark-miki-bucket/output/spark_dummy_data_${rdd.id}.txt")
        rdds.collect().foreach { tweet =>
          log.info(s"key: ${tweet(0)}, polarity: ${tweet(1)}, cleaned text: ${tweet(2)}, created date: ${tweet(3)}")
        }
      }

        // After a period of time, after the calculation is completed, the offset is submitted asynchronously
        inputDStream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
      }

      ssm.start()
      ssm.awaitTermination()
  }
}

