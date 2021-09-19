import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}

object SimpleApp {

  def getAwsCredentials: (String, String) = {
    val accessKeyId = System.getenv("AWS_ACCESS_KEY_ID")
    val secretAccessKey = System.getenv("AWS_SECRET_ACCESS_KEY")
    (accessKeyId, secretAccessKey)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Batch processing").setMaster("spark://spark-master:7077")
    val sc = new SparkContext(conf)

    val log = Logger.getLogger(getClass.getName)
    // specify AWS credentials
    val awsCredentials = getAwsCredentials
    sc.hadoopConfiguration.set("fs.s3a.access.key", awsCredentials._1)
    sc.hadoopConfiguration.set("fs.s3a.secret.key", awsCredentials._2)
    sc.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    sc.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    sc.hadoopConfiguration.set("mapreduce.input.fileinputformat.input.dir.recursive","true")

    val logData = sc.textFile("s3a://test-spark-miki-bucket/output/").cache()
    logData.collect().foreach { tweet =>
      tweet.split(",")
      log.info(s"$tweet")
    }
    sc.stop()
  }
}
