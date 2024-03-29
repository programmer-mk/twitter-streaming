import sbt.Keys.libraryDependencies

name := "twitter-streaming"

version := "0.1"

scalaVersion := "2.11.8"

resolvers += Resolver.sonatypeRepo("releases")


  def dockerSettings(sparkImage: Boolean = false, mainClass: String = "") = Seq(

    dockerfile in docker := {
      val artifactSource: File = assembly.value
      val artifactTargetPath = s"/project/${artifactSource.name}"
      val scriptSourceDir = baseDirectory.value / "../deploy"
      val projectDir = "/project"

      if (sparkImage) {
        new Dockerfile {
          from("bde2020/spark-worker:3.1.1-hadoop3.2")
          add(artifactSource, artifactTargetPath)
          copy(scriptSourceDir, projectDir)
          env("AWS_ACCESS_KEY_ID", "xx")
          env("AWS_SECRET_ACCESS_KEY", "xx")
          env("SPARK_APPLICATION_MAIN_CLASS", mainClass)
          env("ENABLE_INIT_DAEMON", "false")
          run("chmod", "+x", "/project/template.sh")
          run("chmod", "+x", "/project/submit.sh")
          run("chmod", "+x", "/project/test-script.sh")
          //entryPoint(s"/project/test-script.sh")
          entryPoint(s"/project/template.sh")
          cmd(projectDir, s"${name.value}", s"${version.value}")
        }
      } else {
        new Dockerfile {
          from("saumitras01/java:1.8.0_111")
          add(artifactSource, artifactTargetPath)
          copy(scriptSourceDir, projectDir)
          run("chmod", "+x", "/project/start.sh")
          run("chmod", "+x", "/project/wait-for-it.sh")
          entryPoint(s"/project/start.sh")
          cmd(projectDir, s"${name.value}", s"${version.value}")
        }
      }
    },
    imageNames in docker := Seq(
      ImageName(s"mkovacevic/${name.value}:latest")
    )
  )


  lazy val producer = (project in file("tweet-producer"))
    .enablePlugins(sbtdocker.DockerPlugin)
    .settings(
      libraryDependencies ++= Seq(
        "org.scalaj" %% "scalaj-http" % "2.4.2",
        "com.typesafe.play" %% "play-json" % "2.7.3",
        "com.danielasfregola" %% "twitter4s" % "6.2",
        "org.apache.kafka" % "kafka_2.11" % "0.10.0.0" withSources() exclude("org.slf4j", "slf4j-log4j12") exclude("javax.jms", "jms") exclude("com.sun.jdmk", "jmxtools") exclude("com.sun.jmx", "jmxri"),
        "org.apache.kafka" % "kafka-clients" % "2.8.0",
        "org.codehaus.jackson" % "jackson-mapper-asl" % "1.9.13",
        "com.fasterxml.jackson.core" % "jackson-databind" % "2.5.3",
        "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.8.4",
        "com.fasterxml.jackson.datatype" % "jackson-datatype-jsr310" % "2.12.5"
      ),
      dockerSettings()
    )

  lazy val consumer = (project in file("tweet-consumer"))
    .enablePlugins(sbtdocker.DockerPlugin)
    .settings(
      libraryDependencies += "org.apache.kafka" % "kafka_2.11" % "0.10.0.0" withSources() exclude("org.slf4j", "slf4j-log4j12") exclude("javax.jms", "jms") exclude("com.sun.jdmk", "jmxtools") exclude("com.sun.jmx", "jmxri"),
      dockerSettings()
    )

  lazy val sparkStreamingProcessor = (project in file("spark-streaming"))
    .enablePlugins(sbtdocker.DockerPlugin)
    .settings(
      libraryDependencies ++= Seq(
        "org.apache.hadoop" % "hadoop-common" % "3.2.0",
        "org.apache.hadoop" % "hadoop-client" % "3.2.0",
        "org.apache.hadoop" % "hadoop-aws" % "3.2.0",
        "org.apache.spark" %% "spark-core" % "3.1.1" % "provided",
        "org.apache.spark" %% "spark-sql" % "3.1.1" % "provided",
        "org.apache.spark" %% "spark-mllib" % "3.1.1" % "provided",
        "org.apache.spark" %% "spark-streaming" % "3.1.1" % "provided",
        "org.apache.spark" % "spark-streaming-kafka-0-10_2.12" % "3.1.2",
        "org.apache.spark" % "spark-sql-kafka-0-10_2.12" % "3.1.2",
        "org.apache.kafka" %% "kafka" % "2.8.0",
        "org.codehaus.jackson" % "jackson-mapper-asl" % "1.9.13",
        "com.fasterxml.jackson.core" % "jackson-databind" % "2.5.3",
        "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.8.4",
        "com.fasterxml.jackson.datatype" % "jackson-datatype-jsr310" % "2.12.5",

        // for text analysis needed
        "com.github.apanimesh061" % "vader-sentiment-analyzer" % "1.0",
        "org.apache.lucene" % "lucene-analyzers-common" % "6.6.0",

        // mysql connection
        "mysql" % "mysql-connector-java" % "8.0.25"
      ),
      dockerSettings(true, "KafkaStructuredStreaming")
    )

  lazy val sparkBatchProcessor = (project in file("spark-batch"))
    .enablePlugins(sbtdocker.DockerPlugin)
    .settings(
      libraryDependencies ++= Seq(
        "org.apache.hadoop" % "hadoop-common" % "3.2.0",
        "org.apache.hadoop" % "hadoop-client" % "3.2.0",
        "org.apache.hadoop" % "hadoop-aws" % "3.2.0",
        "org.apache.spark" %% "spark-core" % "3.1.1" % "provided",
        "org.apache.spark" %% "spark-mllib" % "3.1.1" % "provided",
        "log4j" % "log4j" % "1.2.14"
      ),
      dockerSettings(true, "TweetPolarityAggregator")
    )

