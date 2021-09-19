import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}
import java.text.SimpleDateFormat

object SimpleApp {

  def getAwsCredentials: (String, String) = {
    val accessKeyId = System.getenv("AWS_ACCESS_KEY_ID")
    val secretAccessKey = System.getenv("AWS_SECRET_ACCESS_KEY")
    (accessKeyId, secretAccessKey)
  }

  def extractDate(date: String): String = {
    val dateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd")
    dateFormat.format(dateFormat.parse(date))
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Batch processing").setMaster("spark://spark-master:7077")
    val sc = new SparkContext(conf)

    //val log = Logger.getLogger(getClass.getName)

    // specify AWS credentials
    val awsCredentials = getAwsCredentials
    sc.hadoopConfiguration.set("fs.s3a.access.key", awsCredentials._1)
    sc.hadoopConfiguration.set("fs.s3a.secret.key", awsCredentials._2)
    sc.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    sc.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive","true")

    val tweetData = sc.textFile("s3a://test-spark-miki-bucket/output/").cache()
    val datePolarityGrouped = tweetData.map { tweet =>
      val tweetMetadata = tweet.split(",")
      val date = tweetMetadata(3)
      // get date and polarity
      (extractDate(date), tweetMetadata(1).toDouble)
    }

    val initialCount = 0.0;
    val addToCounts = (v: Double, n: Double) => n + v
    val sumPartitionCounts = (p1: Double, p2: Double) => p1 + p2

    val aggregatedResults = datePolarityGrouped.aggregateByKey(initialCount)(addToCounts, sumPartitionCounts)

    aggregatedResults.repartition(1).saveAsTextFile(s"s3a://test-spark-miki-bucket/aggregated-results")

    // add logs
    // remove duplicates before aggregation

    sc.stop()
  }
}
