import org.apache.log4j.Logger
import org.apache.spark.sql.{Dataset, SaveMode, SparkSession}
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.functions.{from_json, col}

object KafkaStructuredStreaming {

  val log: Logger = Logger.getLogger(getClass.getName)

  case class UserTweet(id: String, text: String, creationTime: String)

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(getClass.getName)
      .master("spark://spark-master:7077")
      .getOrCreate()

    spark.sparkContext.getConf.set("spark.executor.memory", "1g")
    spark.sparkContext.getConf.set("spark.driver.memory", "1g")
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", System.getenv("AWS_ACCESS_KEY_ID"))
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", System.getenv("AWS_SECRET_ACCESS_KEY"))
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    import spark.implicits._

    val df = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "kafka:9092")
      .option("startingOffsets", "latest") // just messages after spark is up
      .option("subscribe", "tweet-upload-teest")
      .option("group.id", "kafka-spark-integration")
      .load()


    val schema = new StructType()
      .add("id",StringType)
      .add("text",StringType)
      .add("creationTime",StringType)

    val tweetDf = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)").select(from_json(col("value"), schema).as("data"))
      .select("data.*")

    val tweetDataSet = tweetDf.as[UserTweet]

    val stream = tweetDataSet.writeStream
      .outputMode("append")
      .format("csv")
      .option("header", "false")
      .option("sep", ",")
      .option("path", "s3a://test-spark-miki-bucket/output")
      .option("checkpointLocation", "s3a://test-spark-miki-bucket/streaming/checkpoint")
      .trigger(Trigger.ProcessingTime("30 seconds"))
      .start()

    val url="jdbc:mysql://db:3306/test_db"
    val user ="root"
    val password = "root"

    //val writer = new JDBCSink(url,user, password)

//    val query = tweetDataSet.writeStream
//        .format("jdbc")
//        .trigger(Trigger.ProcessingTime("10 seconds"))
//        .foreachBatch((batchDF: Dataset[Tweet], batchId: Long) => {
//        if (!batchDF.isEmpty) {
//          batchDF.coalesce(1)
//            .write
//            .mode(SaveMode.Append)
//            .format("jdbc")
//            .option("driver", "com.mysql.cj.jdbc.Driver")
//            .option("url", url)
//            .option("user", user)
//            .option("password", password)
//            .option("dbtable", "tweets")
//            .save()
//        }
//      }).start()


    stream.awaitTermination()
    //query.awaitTermination()
  }
}
