import org.apache.spark.sql.{SaveMode, SparkSession}

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
      .master("spark://spark-master:7077")
      .getOrCreate()

    // specify AWS credentials
    val awsCredentials = getAwsCredentials
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", awsCredentials._1)
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", awsCredentials._2)
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark.sparkContext.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive","true")

    val tweetData = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("s3a://test-spark-miki-bucket/output")
      .distinct()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    val datePolarityGrouped = tweetData.map(row => {
      val createdDate = row.getTimestamp(1)
      val polarity = row.getDouble(4)
      (createdDate, polarity)
     }
    ).toDF("date","polarity")

    val transformedDf = datePolarityGrouped
      .withColumn("date", to_date($"date", "MM-dd-yyyy"))

    val aggregatedResults = transformedDf.groupBy("date").sum("polarity").toDF("date","total_polarity")

    aggregatedResults
      .repartition(1) // write everything in one file
      .write
      .option("header", "true")
      .csv("s3a://test-spark-miki-bucket/aggregated-results")

    spark.stop()
  }
}
