import sbt.Keys.libraryDependencies

name := "twitter-streaming"

version := "0.1"

scalaVersion := "2.11.8"

resolvers += Resolver.sonatypeRepo("releases")

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.6"
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "2.4.6"
libraryDependencies ++= Dependencies.clientDependencies



def dockerSettings(debugPort: Option[Int] = None) = Seq(

  dockerfile in docker := {
    val artifactSource: File = assembly.value
    val artifactTargetPath = s"/project/${artifactSource.name}"
    val scriptSourceDir = baseDirectory.value / "../deploy"
    val projectDir = "/project/"

    new Dockerfile {
      from("saumitras01/java:1.8.0_111")
      add(artifactSource, artifactTargetPath)
      copy(scriptSourceDir, projectDir)
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