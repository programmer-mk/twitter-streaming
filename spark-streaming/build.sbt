name := "spark-streaming-app"

version := "1.0"

scalaVersion :=  "2.11.8"


//assemblyMergeStrategy in assembly := {
//  case x if x.endsWith("UnusedStubClass.class") => MergeStrategy.last
//  case x if x.endsWith("git.properties") => MergeStrategy.last
//  case x if x.endsWith("Inject.class") => MergeStrategy.last
//  case x if x.endsWith("Named.class") => MergeStrategy.last
//  case x if x.endsWith("Provider.class") => MergeStrategy.last
//  case x if x.endsWith("Qualifier.class") => MergeStrategy.last
//  case x if x.endsWith("package-info.class") => MergeStrategy.last
//  case x if x.endsWith("FastHashMap$Values.class") => MergeStrategy.last
//  case x if x.endsWith("FastHashMap.class") => MergeStrategy.last
//  case x if x.endsWith("FastHashMap$CollectionView.class") => MergeStrategy.last
//  case x if x.endsWith("FastHashMap$CollectionView$CollectionViewIterator.class") => MergeStrategy.last
//  case PathList(ps @ _*) if ps.last endsWith "mime.types" => MergeStrategy.last
//  case x =>
//    val oldStrategy = (assemblyMergeStrategy in assembly).value
//    oldStrategy(x)
//}