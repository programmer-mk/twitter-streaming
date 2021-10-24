package model

import java.util

import org.apache.kafka.common.serialization.Serializer
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule
import producer.UserTweet

class TweetSerializer extends Serializer[UserTweet]{
  override def configure(map: util.Map[String, _], b: Boolean): Unit = {
  }
  override def serialize(s: String, tweet: UserTweet): Array[Byte] = {
    if(tweet==null)
      null
    else
    {
      val objectMapper = new ObjectMapper().registerModule(new JavaTimeModule());
      val bytes = objectMapper.writeValueAsString(tweet).getBytes
      bytes
    }
  }
  override def close(): Unit = {
  }
}