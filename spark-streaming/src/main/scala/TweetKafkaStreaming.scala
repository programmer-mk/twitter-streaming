import com.vader.sentiment.analyzer.SentimentAnalyzer
import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.streaming.dstream.InputDStream
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.{Seconds, StreamingContext}

object TweetKafkaStreaming {
  // Kafka's broker address
  val brokers: String = "kafka:9092"
  // Each stream uses a separate group.id
  val groupId: String = "kafka_spark_straming"
  // The topic to be consumed, accepts an array, and can pass in multiple topics
  val topics: Array[String] = Array("tweets")
  // Used to record logs
  val log: Logger = Logger.getLogger(getClass.getName)

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

    val sparkConf: SparkConf = new SparkConf().setMaster("spark://spark-master:7077").setAppName(getClass.getName)
    val ssm: StreamingContext = new StreamingContext(sparkConf, Seconds(30))
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", "xx")
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", "xxx")
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    ssm.sparkContext.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")


   // @transient
    val inputDStream: InputDStream[ConsumerRecord[String, String]] = getKafkaStream(ssm, topics)
    // this writes to s3
    inputDStream.saveAsTextFiles("s3a://test-spark-miki-bucket/output/spark_streaming-", ".txt")

    //this writes to std output
    inputDStream.foreachRDD { rdd =>
      // Get the offset
      val offsetRanges = rdd.asInstanceOf[HasOffsetRanges].offsetRanges

      if(rdd.count() > 0) {
        log.info(s"rdd size: ${rdd.count()}")
        val rdds: RDD[(String, String)] = rdd.map(x => (x.key(), x.value())).cache()
        //rdds.saveAsTextFile(s"s3a://test-spark-miki-bucket/output/spark_dummy_data_${rdd.id}.txt")
        rdds.collect().foreach{ tweet =>
          log.info(s"key: ${tweet._1}, value: ${tweet._2}")
        }
      }
      // After a period of time, after the calculation is completed, the offset is submitted asynchronously
      inputDStream.asInstanceOf[CanCommitOffsets].commitAsync(offsetRanges)
    }

    ssm.start()
    ssm.awaitTermination()
  }
}

//    val sentimentAnalyzer = new SentimentAnalyzer("I like Serbia")
//    sentimentAnalyzer.analyze()
//    val polarities = sentimentAnalyzer.getPolarity
//    val compoundPolarity = polarities.get("compound")
//    print(compoundPolarity)
