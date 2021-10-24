package producer

import java.time.Instant

class UserTweet {
  private var id: String = ""
  private var text: String = ""
  private var creationTime: Instant = Instant.now
  def this(id: String, text: String, creationTime: Instant) {
    this()
    this.id =id
    this.text = text
    this.creationTime = creationTime
  }
  def getId: String = this.id
  def getText: String = this.text
  def getCreationTime: Instant = this.creationTime
  override def toString: String = "User(" + id + ", " + text + ", " + creationTime + ")"
}
