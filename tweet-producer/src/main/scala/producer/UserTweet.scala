package producer

class UserTweet {
  private var t_key: String = ""
  private var text: String = ""
  private var created: String = ""
  private var searchTerm: String = ""
  def this(t_key: String, text: String, created: String, searchTerm: String) {
    this()
    this.t_key = t_key
    this.text = text
    this.created = created
    this.searchTerm = searchTerm
  }
  def getKey: String = this.t_key
  def getText: String = this.text
  def getCreated: String = this.created
  def getSearchTerm: String = this.searchTerm
  override def toString: String = "User(" + t_key + ", " + text + ", " + created + ", " + searchTerm +  ")"
}
