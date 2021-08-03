//import Dependencies.{akka, akkaHttp}
//import sbt._
//
//object Dependencies extends Build {
//    def akka(artifact: String): ModuleID = "com.typesafe.akka" %% artifact % "2.5.30"
//    def akkaHttp(artifact: String): ModuleID = "com.typesafe.akka" %% artifact % "10.0.5"
//
//    lazy val akkaHttpDependencies = Seq(
//        akkaHttp("akka-http-core"),
//        akkaHttp("akka-http")
//    )
//
//    lazy val akkaDependencies = Seq(
//        akka("akka-actor"),
//        akka("akka-stream")
//    ) ++ akkaHttpDependencies
//
//
//    lazy val clientDependencies = akkaHttpDependencies ++ akkaDependencies
//}
//

