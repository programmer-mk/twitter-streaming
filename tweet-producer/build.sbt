name := "tweet-producer-app"

version := "1.0"

scalaVersion :=  "2.11.8"


assemblyMergeStrategy in assembly := {
  case x if x.endsWith("public-suffix-list.txt") => MergeStrategy.first
  case x if x.endsWith("module-info.class") => MergeStrategy.discard
  case x if x.endsWith("io.netty.versions.properties") => MergeStrategy.first
  case x if x.endsWith("UnusedStubClass.class") => MergeStrategy.first
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}