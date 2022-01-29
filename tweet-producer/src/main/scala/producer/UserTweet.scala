package producer


case class TweetStats(likeCount: Int, retweetCount: Int, userFollowersCount: Int)

class UserTweet {
  private var t_key: String = ""
  private var text: String = ""
  private var created: String = ""
  private var searchTerm: String = ""
  private var stats: TweetStats =  TweetStats(0,0,0)
  def this(t_key: String, text: String, created: String, searchTerm: String, stats: TweetStats) {
    this()
    this.t_key = t_key
    this.text = text
    this.created = created
    this.searchTerm = searchTerm
    this.stats = stats
  }
  def getKey: String = this.t_key
  def getText: String = this.text
  def getCreated: String = this.created
  def getSearchTerm: String = this.searchTerm
  def getStats: TweetStats = this.stats
  override def toString: String = "User(" + t_key + ", " + text +
    ", " + created + ", " + searchTerm +  ", "+ stats.likeCount + ", " + stats.retweetCount +  ", " +  stats.userFollowersCount + ")"
}
