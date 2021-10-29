import java.sql._

import org.apache.spark.sql.ForeachWriter

class  JDBCSink(url:String, user:String, pwd:String) extends ForeachWriter[(String, String)] {
  val driver = "com.mysql.cj.jdbc.Driver"
  var connection:Connection = _
  var statement:Statement = _

  def open(partitionId: Long,version: Long): Boolean = {
    Class.forName(driver)
    connection = DriverManager.getConnection(url, user, pwd)
    statement = connection.createStatement
    true
  }

  def process(value: (String, String)): Unit = {
    statement.executeUpdate("insert into tweets(t_key,text,processed_text,created) values('" + value._1 + "','" + value._2 + "','" + "test" + "','" + "test"+ "')")
  }

  def close(errorOrNull: Throwable): Unit = {
    connection.close
  }
}