package producer

class UserTweet {
  private var t_key: String = ""
  private var text: String = ""
  private var created: String = ""
  def this(t_key: String, text: String, created: String) {
    this()
    this.t_key = t_key
    this.text = text
    this.created = created
  }
  def getKey: String = this.t_key
  def getText: String = this.text
  def getCreated: String = this.created
  override def toString: String = "User(" + t_key + ", " + text + ", " + created + ")"
}
