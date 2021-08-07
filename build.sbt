import sbt.Keys.libraryDependencies

name := "twitter-streaming"

version := "0.1"

scalaVersion := "2.11.8"

resolvers += Resolver.sonatypeRepo("releases")


def dockerSettings(debugPort: Option[Int] = None) = Seq(

  dockerfile in docker := {
    val artifactSource: File = assembly.value
    val artifactTargetPath = s"/project/${artifactSource.name}"
    val scriptSourceDir = baseDirectory.value / "../deploy"
    val projectDir = "/project"

    new Dockerfile {
      from("saumitras01/java:1.8.0_111")
      add(artifactSource, artifactTargetPath)
      copy(scriptSourceDir, projectDir)
      run("chmod", "+x", "/project/start.sh")
      run("chmod", "+x", "/project/wait-for-it.sh")
      entryPoint(s"/project/start.sh")
      cmd(projectDir, s"${name.value}", s"${version.value}")
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
      "org.apache.kafka" % "kafka_2.11" % "0.10.0.0" withSources() exclude("org.slf4j","slf4j-log4j12") exclude("javax.jms", "jms") exclude("com.sun.jdmk", "jmxtools") exclude("com.sun.jmx", "jmxri")
    ),
    dockerSettings()
  )

lazy val consumer = (project in file("tweet-consumer"))
  .enablePlugins(sbtdocker.DockerPlugin)
  .settings(
    libraryDependencies += "org.apache.kafka" % "kafka_2.11" % "0.10.0.0" withSources() exclude("org.slf4j","slf4j-log4j12") exclude("javax.jms", "jms") exclude("com.sun.jdmk", "jmxtools") exclude("com.sun.jmx", "jmxri"),
    dockerSettings()
  )

lazy val sparkStreamingProcessor = (project in file("spark-streaming"))
  .enablePlugins(sbtdocker.DockerPlugin)
  //.settings(
   // libraryDependencies ++= Seq(
//      "org.apache.spark" %% "spark-sql" % "2.4.6" % "provided",
//      "org.apache.spark" %% "spark-streaming" % "2.4.6" % "provided",
//     // "org.apache.spark" %% "org.apache.spark" % "spark-streaming-kafka" % "1.6.0",
//      "org.apache.spark" % "spark-streaming-kafka-0-10_2.11" % "2.2.0",
//
//      "com.typesafe.akka" %% "akka-actor" % "2.5.30",
//      "com.typesafe.akka" %% "akka-stream" % "2.5.30",
//      "com.typesafe.akka" %% "akka-http-core" % "10.0.5",
//      "com.typesafe.akka" %% "akka-http" % "10.0.5"
   // )
    //dockerSettings()
//  )


