package producer


import java.util.Properties

import akka.actor.ActorSystem
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import play.api.libs.json.{JsValue, Json}
import scalaj.http.{Http, HttpOptions}

import scala.concurrent.ExecutionContext

object TwitterProducer {

  def streamTweetsAPI(props: Properties, TOPIC: String): Unit = {
    val searchTerms = Seq("microsoft", "amazon", "bitcoin", "apple", "tesla")
    val producer = new KafkaProducer[String, UserTweet](props)
    try {
      searchTerms foreach  { term =>
        var stillWork = true
        var token = ""
        while(stillWork) {
          val queryJson = if (token.isEmpty) {
            s"""{
              "query":"${term} lang:en",
              "maxResults": "100",
              "fromDate":"201802010000",
              "toDate":"201802012359"
            }""".stripMargin
          } else {
            s"""{
              "query":"oculus lang:en",
              "maxResults": "100",
               "fromDate":"201802010000",
              "toDate":"201802012359",
              "next": "${token}"
            }""".stripMargin
          }
          val response = Http("https://api.twitter.com/1.1/tweets/search/fullarchive/prod.json")
            .postData(queryJson)
            .header("Content-Type", "application/json")
            .header("authorization", "Bearer AAAAAAAAAAAAAAAAAAAAAOmyRgEAAAAA4YesubUxyPH%2FtHMbtIlb0JAcu0A%3DBSNh2XxAXd6rUzZxS98TB10O9fozMTzxd2ASinikviIHKdG5L9")
            .option(HttpOptions.readTimeout(10000)).asString

          val body = response.body
          val json: JsValue = Json.parse(body)
          val results = (json \ "results").as[Seq[JsValue]].map { tweet =>
            val idStr = (tweet \ "id_str").as[String]
            val text = (tweet \ "text").as[String]
            val createdAt = (tweet \ "created_at").as[String]
            val likeCount = (tweet \ "favorite_count").as[Int]
            val retweetCount = (tweet \ "retweet_count").as[Int]
            val user = (tweet \ "user").as[JsValue]
            val userFollowersCount = (user \ "followers_count").as[Int]
            new UserTweet(idStr, text, createdAt, term, TweetStats(likeCount, retweetCount, userFollowersCount))
          }

          results.foreach { tweet =>
            val data = new ProducerRecord[String, UserTweet](TOPIC, tweet.getKey, tweet)
            println(tweet.toString)
            producer.send(data)
          }

          if(results.size < 100) {
            // arrived to end of the list
            stillWork = false
          }else {
            token = (json \ "next").as[String]
          }
          Thread.sleep(5000)
        }
      }
    } catch {
      case e: java.util.concurrent.TimeoutException =>
        Thread.sleep(5000)
        println(s"Timeout occured. Sleeping for 5 seconds.Error: $e")
      case e: Exception =>
        Thread.sleep(15000)
        println(s"Error retrieving tweet. Check tweet streaming endpoint connection.Error: $e")
    }
  }

  def readTweetsFromDisk(props: Properties, TOPIC: String): Unit = {
  //def readTweetsFromDisk(): Unit = {
    //val bufferedSource = io.Source.fromFile("./tweet-producer/Company_Tweet.csv")
    //val bufferedSource2 = io.Source.fromFile("./tweet-producer/Tweet_Preprocessed.csv")

    println("Start reading tweets from disk!")
    val bufferedSource = scala.io.Source.fromInputStream(getClass.getResourceAsStream("/Company_Tweet.csv"))
    val bufferedSource2 = scala.io.Source.fromInputStream(getClass.getResourceAsStream("/Tweet_Preprocessed.csv"))
    println("Tweets loaded into memory!")

    var tweetCompanyMapper = Map[String, String]()
    bufferedSource.getLines.foreach { line =>
      val cols = line.split(",").map(_.trim)
      val company = if (cols(1) == "GOOGL") {
        "GOOG"
      } else {
        cols(1)
      }
      tweetCompanyMapper += (cols(0) -> company)
    }
    bufferedSource.close
    println("Stock map created!")

    val producer = new KafkaProducer[String, UserTweet](props)

    //tweet_id,writer,post_date,body,comment_num,retweet_num,like_num
    bufferedSource2.getLines.filter(line => !line.contains("tweet_id")).toSeq.foreach { line =>
      val cols = line.split(",").map(_.trim)
      val term = tweetCompanyMapper.get(cols(0))
      val tweet = new UserTweet(cols(0), cols(3), cols(2), term.getOrElse(""), TweetStats(cols(6).toInt, cols(5).toInt, 0))
      val data = new ProducerRecord[String, UserTweet](TOPIC, tweet.getKey, tweet)
      println(tweet.toString)
      Thread.sleep(1000)
      producer.send(data)
      //tweets += tweet
    }
    bufferedSource2.close
    println("Tweets extracted!")
  }

  def main(args : Array[String]) {
    // get an ActorSystem in scope for the futures
    implicit val system = ActorSystem("TwitterFutureSystem")
    implicit val ec: ExecutionContext = system.dispatcher
    //val BROKER_LIST = "kafka1:9092,kafka2:9093,kafka3:9094" //change it to localhost:9092 if not connecting through docker
    val BROKER_LIST = System.getenv("KAFKA_SERVICE")
    val TOPIC = "tweet-upload-teest"
    val fromDate = System.getenv("FROM_DATE")
    val toDate = System.getenv("TO_DATE")

    val props = new Properties()
    props.put("bootstrap.servers", BROKER_LIST)
    props.put("client.id", "KafkaTweetProducer")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "model.TweetSerializer")

    val test = true
    if(test) {
      readTweetsFromDisk(props, TOPIC)
      //readTweetsFromDisk()
    }else {
      // small request limit
      //streamTweetsAPI(props, TOPIC)
    }
  }
}