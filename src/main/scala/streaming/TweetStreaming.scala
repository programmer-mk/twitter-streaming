package streaming

import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}


object TweetStreaming {
  def main(args: Array[String]): Unit = {
    //Create SparkSession
    val spark: SparkSession = SparkSession.builder().appName("TwitterStreaming").master("local[3]").getOrCreate()

    //Set the log level
    spark.sparkContext.setLogLevel("WARN")

    //Read the latest data The data in socketDatas is Row type
    //host: the IP address of the connected server
    //port: the port number of the connection
    val socketDatas: DataFrame = spark.readStream.format("socket")
      .option("host", "localhost")
      .option("port", "8080")
      .load()


    //Data processing according to business logic
    //Convert the data type because the data type of socketDatas is Row, we need to convert it to String type to do data processing
    import spark.implicits._
    val strDatas: Dataset[String] = socketDatas.as[String]
    //Process data to find the number of words (WordCount)
    val resDatas: Dataset[Row] = strDatas.flatMap(x => x.split(" ")).groupBy("value").count().sort($"count".desc)


    //Calculation result output
    //format: indicates where the data is output
    //outputMode: which data to output
    //trigger: trigger
    //start: start the task
    //awaitTermination: waiting to close
    resDatas.writeStream
      .format("console")
      .outputMode("complete")
      .trigger(Trigger.ProcessingTime(1000))
      .start()
      .awaitTermination()


  }
}
