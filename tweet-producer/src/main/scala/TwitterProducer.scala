import java.util.Properties
import java.util.concurrent.TimeUnit

import akka.actor.ActorSystem
import com.danielasfregola.twitter4s.TwitterRestClient
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}

import scala.concurrent.duration.{Duration, FiniteDuration}
import scala.concurrent.{Await, ExecutionContext, Future}

object TwitterProducer {

  def main(args : Array[String]) {
    // get an ActorSystem in scope for the futures
    implicit val system = ActorSystem("TwitterFutureSystem")
    implicit val ec: ExecutionContext = system.dispatcher
    val BROKER_LIST = "kafka:9092" //change it to localhost:9092 if not connecting through docker
    val TOPIC = "tweets"

    val twitterClient = TwitterRestClient()
    val nums = (1 to 10).toList

    val props = new Properties()
    props.put("bootstrap.servers", BROKER_LIST)
    props.put("client.id", "KafkaTweetProducer")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")

    val producer = new KafkaProducer[String, String](props)

    while(true) {
      val tweets = Future.sequence(nums.map { _ =>
        val searchTweets = twitterClient.searchTweet("football")
        searchTweets
      })

      val maxWaitTime: FiniteDuration = Duration(5, TimeUnit.SECONDS)
      val completedResults = Await.result(tweets, maxWaitTime)
      completedResults.flatMap(_.data.statuses) foreach { tweet =>
        val data = new ProducerRecord[String, String](TOPIC, tweet.text)
        producer.send(data)
      }
      println(completedResults)
    }
  }
}
