import akka.actor.ActorSystem
import com.danielasfregola.twitter4s.TwitterRestClient
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, ExecutionContext, Future}


object TwitterConsumer {

  def main(args : Array[String]) {
    // get an ActorSystem in scope for the futures
    implicit val system = ActorSystem("TwitterFutureSystem")
    implicit val ec: ExecutionContext = system.dispatcher

    val nums = Seq(1,2,3)

    val twitterClient = TwitterRestClient()

    val tweets = Future.sequence(nums.map { _ =>
      val searchTweets = twitterClient.searchTweet("football")
      searchTweets
    })

    val completedResults = Await.result(tweets, Duration("5"))
    println(completedResults)
  }
}