package producer

import java.time.Instant
import java.util.{Properties, Random}
import java.util.concurrent.TimeUnit

import akka.actor.ActorSystem
import com.danielasfregola.twitter4s.TwitterRestClient
import com.danielasfregola.twitter4s.entities.enums.Language
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import play.api.libs.json.{JsValue, Json}
import scalaj.http.{Http, HttpOptions}

import scala.concurrent.duration.{Duration, FiniteDuration}
import scala.concurrent.{Await, ExecutionContext}
import scala.util.parsing.json.JSON

object TwitterProducer {

  def main(args : Array[String]) {
    // get an ActorSystem in scope for the futures
    implicit val system = ActorSystem("TwitterFutureSystem")
    implicit val ec: ExecutionContext = system.dispatcher
    //val BROKER_LIST = "kafka1:9092,kafka2:9093,kafka3:9094" //change it to localhost:9092 if not connecting through docker
    val BROKER_LIST = System.getenv("KAFKA_SERVICE")
    val TOPIC = "tweet-upload-teest"

    val searchTerms = Seq("microsoft", "amazon", "bitcoin", "apple", "tesla")

    val props = new Properties()
    props.put("bootstrap.servers", BROKER_LIST)
    props.put("client.id", "KafkaTweetProducer")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "model.TweetSerializer")

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
              "toDate":"201802282359"
            }""".stripMargin
          } else {
            s"""{
              "query":"${term} lang:en",
              "maxResults": "100",
              "fromDate":"201802010000",
              "toDate":"201802282359",
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
            new UserTweet(idStr, text, createdAt, term)
          }

          results.foreach { tweet =>
            val data = new ProducerRecord[String, UserTweet](TOPIC, tweet.getKey, tweet)
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
}