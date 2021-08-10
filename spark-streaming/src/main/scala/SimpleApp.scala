/* SimpleApp.scala */
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

    // specify AWS credentials
    val awsCredentials = getAwsCredentials
    sc.hadoopConfiguration.set("fs.s3n.access.key", awsCredentials._1)
    sc.hadoopConfiguration.set("fs.s3n.secret.key", awsCredentials._2)
    sc.hadoopConfiguration.set("fs.s3n.endpoint", "s3.amazonaws.com")
    sc.setLogLevel("ERROR")

    val logData = sc.textFile("s3a://test-spark-miki-bucket/spark_dummy_data.txt").cache()
    val numAs = logData.filter(line => line.contains("a")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println(s"Lines with a: $numAs, Lines with b: $numBs")
    sc.stop()
  }
}