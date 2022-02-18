import org.apache.spark.sql.{SaveMode, SparkSession, functions}

object TweetPolarityAggregator {

  def getAwsCredentials: (String, String) = {
    val accessKeyId = System.getenv("AWS_ACCESS_KEY_ID")
    val secretAccessKey = System.getenv("AWS_SECRET_ACCESS_KEY")
    (accessKeyId, secretAccessKey)
  }

  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName(getClass.getName)
      .master(System.getenv("SPARK_MASTER"))
      .getOrCreate()

    // specify AWS credentials
    val awsCredentials = getAwsCredentials
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", awsCredentials._1)
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", awsCredentials._2)
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark.sparkContext.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive","true")

    val s3StreamingOutputKey = System.getenv("TWEETS_STREAMING_OUTPUT")
    val s3AggregatedOutputKey = System.getenv("TWEETS_AGGREGATED_OUTPUT")

    val tweetData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(s3StreamingOutputKey)
      .distinct()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    val datePolarityGrouped = tweetData.map(row => {
      val t_key = row.getLong(0)
      val createdDate = row.getInt(1)
      val polarity = row.getDouble(5)
      val searchTerm = row.getString(2)
      (t_key, createdDate, searchTerm ,polarity)
     }
    ).toDF("t_key", "date","search_term", "polarity")
      .withColumn("date", to_timestamp(from_unixtime($"date")))

    val transformedDf = datePolarityGrouped
      .withColumn("date", to_date($"date", "MM-dd-yyyy"))

    val aggregatedResults = transformedDf.groupBy("date", "search_term").agg(
      functions.count("t_key").as("count"),
      functions.sum("polarity").as("agg_polarity"),
      functions.count(functions.when(col("polarity").gt(0), 1)).as("positive_count"),
      functions.count(functions.when(col("polarity").lt(0), 1)).as("negative_count"),
      functions.count(functions.when(col("polarity").equalTo(0), 1)).as("neutral_count"),
    )
      //.sum("polarity")
      //.toDF("date", "search_term", "total_polarity", "tweet_count")

    aggregatedResults
      .repartition(1) // write everything in one file
      .write
      .option("header", "true")
      .csv(s3AggregatedOutputKey)

    spark.stop()
  }
}
