package producer

import java.time.Instant
import java.util.Properties
import java.util.concurrent.TimeUnit

import akka.actor.ActorSystem
import com.danielasfregola.twitter4s.TwitterRestClient
import com.danielasfregola.twitter4s.entities.enums.Language
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}

import scala.concurrent.duration.{Duration, FiniteDuration}
import scala.concurrent.{Await, ExecutionContext}

object TwitterProducer {

  def main(args : Array[String]) {
    // get an ActorSystem in scope for the futures
    implicit val system = ActorSystem("TwitterFutureSystem")
    implicit val ec: ExecutionContext = system.dispatcher
    val BROKER_LIST = "kafka:9092" //change it to localhost:9092 if not connecting through docker
    val TOPIC = "tweet-upload-teest"

    val twitterClient = TwitterRestClient()

    val props = new Properties()
    props.put("bootstrap.servers", BROKER_LIST)
    props.put("client.id", "KafkaTweetProducer")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "model.TweetSerializer")

    val producer = new KafkaProducer[String, UserTweet](props)
    val test = false
    if(test) {
      var i = 0
      while(true) {
        val value = s"record-${i}"
        val key = s"key-${i}"
        val dummyTweet = new UserTweet(key, value, Instant.now())
        val data = new ProducerRecord[String, UserTweet](TOPIC, key, dummyTweet)
        i +=1
        Thread.sleep(1000)
        producer.send(data)
        println(value)
      }
    } else {
      while(true) {
        try {
          val searchTweets = twitterClient.searchTweet("microsoft", language = Some(Language.English))
          val maxWaitTime: FiniteDuration = Duration(5, TimeUnit.SECONDS)
          val completedResults = Await.result(searchTweets, maxWaitTime)
          completedResults.data.statuses foreach { tweet =>
            val userTweet = new UserTweet(tweet.id_str, tweet.text, tweet.created_at)
            println(s"tweet value is: $userTweet.text")
            val data = new ProducerRecord[String, UserTweet](TOPIC, tweet.id_str, userTweet)
            producer.send(data)
          }
          Thread.sleep(3000)
        } catch {
          case e: java.util.concurrent.TimeoutException => println(s"Timeout occured. Sleeping for 1 second.Error: $e")
          case e: Exception => println(s"Error retrieving tweet. Check tweet streaming endpoint connection.Error: $e")
        }
      }
    }
  }
}