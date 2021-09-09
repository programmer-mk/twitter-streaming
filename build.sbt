import sbt.Keys.libraryDependencies

name := "twitter-streaming"

version := "0.1"

scalaVersion := "2.11.8"

resolvers += Resolver.sonatypeRepo("releases")


  def dockerSettings(sparkImage: Boolean = false) = Seq(

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
          env("AWS_ACCESS_KEY_ID", "xxxx")
          env("AWS_SECRET_ACCESS_KEY", "xxxx")
          env("SPARK_APPLICATION_MAIN_CLASS", "TweetKafkaStreaming")
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
        "com.danielasfregola" %% "twitter4s" % "6.2",
        "org.apache.kafka" % "kafka_2.11" % "0.10.0.0" withSources() exclude("org.slf4j", "slf4j-log4j12") exclude("javax.jms", "jms") exclude("com.sun.jdmk", "jmxtools") exclude("com.sun.jmx", "jmxri")
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
        "org.apache.spark" %% "spark-mllib" % "3.1.1" % "provided",
        "org.apache.spark" %% "spark-streaming" % "3.1.1" % "provided",
        "org.apache.spark" % "spark-streaming-kafka-0-10_2.12" % "3.1.2",
        "org.apache.kafka" %% "kafka" % "2.8.0",
        "log4j" % "log4j" % "1.2.14",

        // for text analysis needed
        "com.github.apanimesh061" % "vader-sentiment-analyzer" % "1.0",
        "org.apache.lucene" % "lucene-analyzers-common" % "6.6.0"
      ),
      dockerSettings(true)
    )

