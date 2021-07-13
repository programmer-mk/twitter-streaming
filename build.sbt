name := "twitter-streaming"

version := "0.1"

scalaVersion := "2.12.11"

resolvers += Resolver.sonatypeRepo("releases")

libraryDependencies += "com.danielasfregola" %% "twitter4s" % "7.0"

libraryDependencies ++= Dependencies.clientDependencies