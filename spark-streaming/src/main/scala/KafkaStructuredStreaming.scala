import com.vader.sentiment.analyzer.SentimentAnalyzer
import org.apache.log4j.Logger
import org.apache.spark.sql.{Dataset, SaveMode, SparkSession}
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.functions.{col, from_json, lit}
import streaming.Util

object KafkaStructuredStreaming {

  val log: Logger = Logger.getLogger(getClass.getName)

  case class UserTweet(id: Long, t_key: String, processed_text: String, polarity: Double, created: String, search_term: String)

  def computePolarity = (input: String) => {
    val sentimentAnalyzer = new SentimentAnalyzer(input)
    sentimentAnalyzer.analyze()
    val polarities = sentimentAnalyzer.getPolarity
    val compoundPolarity = polarities.get("compound")
    compoundPolarity
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(getClass.getName)
      .master(System.getenv("SPARK_MASTER"))
      .getOrCreate()

    //spark.sparkContext.getConf.set("spark.executor.memory", "1g")
    //spark.sparkContext.getConf.set("spark.driver.memory", "2g")
    //spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", System.getenv("AWS_ACCESS_KEY_ID"))
    //spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", System.getenv("AWS_SECRET_ACCESS_KEY"))
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    import spark.implicits._

    val df = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", System.getenv("KAFKA_SERVICE"))
      .option("startingOffsets", "latest") // just messages after spark is up
      .option("subscribe", "tweet-upload-teest")
      .option("group.id", "kafka-spark-integration")
      .load()

    val schema = new StructType()
      .add("key",StringType)
      .add("text",StringType)
      .add("created",StringType)
      .add("searchTerm",StringType)

    val computePolarityUdf = spark.udf.register("computePolarity", computePolarity)
    val cleanTextUdf = spark.udf.register("cleanTextUdf", Util.cleanDocument)

    val tweetDataset = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
      .select(col("key").as("t_key"), from_json(col("value"), schema).as("data"))
      .select(col("t_key"), col("data.text"), col("data.created"), col("data.searchTerm").as("search_term"))
      .withColumn("id", lit(0))
      .withColumn("processed_text", cleanTextUdf(col("text")))
      .withColumn("polarity", computePolarityUdf(col("processed_text")))
      .drop("text")
      .as[UserTweet]

    val stream = tweetDataset.writeStream
      .outputMode("append")
      .format("csv")
      .option("header", "true")
      .option("path", "s3a://test-spark-miki-bucket/output")
      .option("checkpointLocation", "s3a://test-spark-miki-bucket/streaming/checkpoint")
      .trigger(Trigger.ProcessingTime("30 seconds"))
      .start()
                        // mysql is container name
    val url=s"jdbc:mysql://${System.getenv("MYSQL_SERVICE")}/myDb"
    val user ="myDbUser"
    val password = "myPassword123"

    val query = tweetDataset.writeStream
        .format("jdbc")
        .trigger(Trigger.ProcessingTime("10 seconds"))
        .foreachBatch((batchDF: Dataset[UserTweet], batchId: Long) => {
        if (!batchDF.isEmpty) {
          batchDF.coalesce(1)
            .write
            .mode(SaveMode.Append)
            .format("jdbc")
            .option("driver", "com.mysql.cj.jdbc.Driver")
            .option("url", url)
            .option("user", user)
            .option("password", password)
            .option("dbtable", "tweets")
            .save()
        }
      }).start()

    stream.awaitTermination()
    query.awaitTermination()
  }
}
