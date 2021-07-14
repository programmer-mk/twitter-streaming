import java.util.concurrent.TimeUnit

import akka.actor.ActorSystem
import com.danielasfregola.twitter4s.TwitterRestClient

import scala.concurrent.duration.{Duration, FiniteDuration}
import scala.concurrent.{Await, ExecutionContext, Future}


object TwitterConsumer {

  def main(args : Array[String]) {
    // get an ActorSystem in scope for the futures
    implicit val system = ActorSystem("TwitterFutureSystem")
    implicit val ec: ExecutionContext = system.dispatcher
    val twitterClient = TwitterRestClient()
    val nums = (1 to 10).toList

    while(true) {
      val tweets = Future.sequence(nums.map { _ =>
        val searchTweets = twitterClient.searchTweet("football")
        searchTweets
      })

      val maxWaitTime: FiniteDuration = Duration(5, TimeUnit.SECONDS)
      val completedResults = Await.result(tweets, maxWaitTime)
      println(completedResults)
    }
  }
}