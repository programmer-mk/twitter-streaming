/* SimpleApp.scala */
import org.apache.log4j.Logger
import org.apache.spark.{SparkConf, SparkContext}

object SimpleApp {

  def getAwsCredentials: (String, String) = {
    val accessKeyId = System.getenv("AWS_ACCESS_KEY_ID")
    val secretAccessKey = System.getenv("AWS_SECRET_ACCESS_KEY")
    (accessKeyId, secretAccessKey)
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("spark://spark-master:7077")
    val sc = new SparkContext(conf)

    val log = Logger.getLogger(getClass.getName)
    // specify AWS credentials
    val awsCredentials = getAwsCredentials
    sc.hadoopConfiguration.set("fs.s3a.access.key", awsCredentials._1)
    sc.hadoopConfiguration.set("fs.s3a.secret.key", awsCredentials._2)
    sc.hadoopConfiguration.set("fs.s3a.endpoint", "s3.eu-west-2.amazonaws.com")
    sc.hadoopConfiguration.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    val logData = sc.textFile("s3a://test-spark-miki-bucket/spark_dummy_data.txt").cache()
    logData.filter(line => line.contains("a")).collect().foreach(log.info)
    logData.filter(line => line.contains("b")).collect().foreach(log.info)
    sc.stop()
  }
}