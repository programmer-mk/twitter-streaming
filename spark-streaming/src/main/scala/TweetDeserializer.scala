import java.util

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import org.apache.kafka.common.serialization.Deserializer

class TweetDeserializer extends Deserializer[UserTweet] {
  override def configure(map: util.Map[String, _], b: Boolean): Unit = {
  }
  override def deserialize(s: String, bytes: Array[Byte]): UserTweet = {
    val mapper = new ObjectMapper().registerModule(new JavaTimeModule());
    val user = mapper.readValue(bytes, classOf[UserTweet])
    user
  }
  override def close(): Unit = {
  }
}