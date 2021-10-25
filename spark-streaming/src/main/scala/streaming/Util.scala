package streaming

import org.apache.spark.ml.feature.StopWordsRemover

import scala.io.Source

class Util {

}

object Util {

  val RegexMap = Map[String, String](
    "punctuation" -> "[^a-zA-Z0-9]",
    "digits" -> "\\b\\d+\\b",
    "whiteSpace" -> "\\s+",
    "smallWords" -> "\\b[a-zA-Z0-9]{1,2}\\b",
    "urls" -> "(https?\\://)\\S+"
  )

  val StopWords = Map[String, List[String]](
    "english" ->scala.io.Source.fromInputStream(getClass.getResourceAsStream("/stopwords.txt")).getLines().toList
  )

  def removeRegex(txt: String, flag: String): String = {
    val regex = RegexMap.get(flag)
    var cleaned = txt
    regex match {
      case Some(value) =>
        if (value.equals("whiteSpace")) cleaned = txt.replaceAll(value, "")
        else cleaned = txt.replaceAll(value, " ")
      case None => println("No regex flag matched")
    }
    cleaned
  }

  def removeCustomWords(txt: String, flag: String): String = {
    var words = txt.split(" ")
    val stopwords = StopWords.get(flag)
    stopwords match {
      case Some(value) => words = words.filter(x => !value.contains(x))
      case None => println("No stopword flag matched")
    }
    words.mkString(" ")
  }

  def cleanDocument= (documentText: String) => {
    //  Converting all words to lowercase
    //  Removing URLs from document
    //  Removing Punctuations from document text
    //  Removing Digits from document text
    //  Removing all words with length less than or equal to 2
    //  Removing extra whitespaces from text
    //  Removing English Stopwords
    //  Returning the preprocessing and cleaned document text

    val text = documentText.toLowerCase
    val processings = Seq(
      "urls", "punctuation", "digits", "smallWords", "whiteSpace", "english"
    )
    val cleaned = processings.fold(text) { (technique, preprocessed) =>
      if (technique == "english") {
        removeCustomWords(technique, preprocessed)
      } else {
        removeRegex(technique, preprocessed)
      }
    }
    cleaned
  }

}