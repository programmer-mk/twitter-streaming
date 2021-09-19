package producer

import java.util.Properties
import java.util.concurrent.TimeUnit

import akka.actor.ActorSystem
import com.danielasfregola.twitter4s.TwitterRestClient
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}

import scala.concurrent.duration.{Duration, FiniteDuration}
import scala.concurrent.{Await, ExecutionContext}

object TwitterProducer {

  def main(args : Array[String]) {
    // get an ActorSystem in scope for the futures
    implicit val system = ActorSystem("TwitterFutureSystem")
    implicit val ec: ExecutionContext = system.dispatcher
    val BROKER_LIST = "kafka:9092" //change it to localhost:9092 if not connecting through docker
    val TOPIC = "tweets-test2"

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
        val data = new ProducerRecord[String, String](TOPIC, key, value)
        i +=1
        Thread.sleep(1000)
        //producer.send(data)
        println(value)
      }
    } else {
      while(true) {
        val searchTweets = twitterClient.searchTweet("microsoft")
        val maxWaitTime: FiniteDuration = Duration(5, TimeUnit.SECONDS)
        val completedResults = Await.result(searchTweets, maxWaitTime)
        completedResults.data.statuses foreach { tweet =>
          println(s"TweetId is: ${tweet.id}")

          //producer.send(data)
          println(s"TweetId text value: ${tweet.text}")
          val userTweet = new UserTweet(tweet.id_str, tweet.text, tweet.created_at)
          println(s"tweet value is: $userTweet.text")
          val data = new ProducerRecord[String, UserTweet](TOPIC, tweet.id_str, userTweet)
          producer.send(data)
        }
        Thread.sleep(20000)
      }
    }
  }
}