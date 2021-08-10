name := "spark-streaming-app"

version := "1.0"

scalaVersion :=  "2.12.3"


//libraryDependencies ++= Seq(
//  "org.apache.spark" %% "spark-sql" % "2.4.5" % "provided",
//  "org.apache.spark" %% "spark-streaming" % "2.4.5" % "provided",
//  // "org.apache.spark" %% "org.apache.spark" % "spark-streaming-kafka" % "1.6.0",
//  "org.apache.spark" % "spark-streaming-kafka-0-10_2.11" % "2.2.0" % "provided",
//  "org.apache.hadoop" % "hadoop-aws" % "2.8.2" % "provided",
//  "com.amazonaws" % "aws-java-sdk-pom" % "1.10.34" % "provided"
//)

//libraryDependencies ++= Seq(
//  "org.apache.hadoop" % "hadoop-common" % "3.0.0",
//  "org.apache.hadoop" % "hadoop-client" % "3.0.0",
//  "org.apache.hadoop" % "hadoop-aws" % "3.0.0",
//  "org.apache.spark" %% "spark-core" % "2.4.5" % "provided",
//)

//commonExcludeDependencies ++= Seq(
//  SbtExclusionRule("org.apache.avro","avro-tools"),
//)

assemblyMergeStrategy in assembly := {
  case x if x.endsWith("public-suffix-list.txt") => MergeStrategy.discard
  case x if x.endsWith("module-info.class") => MergeStrategy.discard
  case x =>
    val oldStrategy = (assemblyMergeStrategy in assembly).value
    oldStrategy(x)
}