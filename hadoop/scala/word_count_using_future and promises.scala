import scala.concurrent._
import ExecutionContext.Implicits.global
import scala.util.{ Success, Failure }
import java.io.File
import scala.io.Source

class ProcessFile {

  def scanFiles(docRoot: String): Future[Seq[String]] =
    Future {
      new File(docRoot).list.map(docRoot + _)
    }

  def processFiles(fileNames: Seq[String]): Future[Seq[(String, Int)]] = {
    val futures: Seq[Future[(String, Int)]] = fileNames.map(name => processFile(name))
    val singleFuture: Future[Seq[(String, Int)]] = Future.sequence(futures)
    singleFuture.map(r => r.sortWith(_._2 < _._2))
  }

  def processFile(fileName: String): Future[(String, Int)] =
    Future {
      val dataFile = new File(fileName)
      val wordCount = Source.fromFile(dataFile).getLines.foldRight(0)(_.split(" ").size + _)
      (fileName, wordCount)
    } recover {
      case e: java.io.IOException =>
        println("Something went wrong " + e)
        (fileName, 0)
    }

}

class PromiseConstruct() {

  def computeWordCount(path: String): Future[Seq[(String, Int)]] = {
    //Create a promise
    val promiseOfFinalResult = Promise[Seq[(String, Int)]]()
    val processFile = new ProcessFile()
    val futureWithResult: Future[Seq[(String, Int)]] = for {
      files <- processFile.scanFiles(path)
      result <- processFile.processFiles(files)
    } yield {
      result
    }

    //complete the Promise by setting a successful value -- can't be changed once set
    futureWithResult.onSuccess { case r => promiseOfFinalResult.success(r) }

    //return the read side of the Promise
    promiseOfFinalResult.future
  }
}

object FutureWithPromise1 {
  def main(args: Array[String]) {
    val path = "C:\\Users\\Deepak\\Desktop\\test_scala_wordcount\\"
    val promise = new PromiseConstruct()
    val wordCount = promise.computeWordCount(path)

    //use the future using callback
    wordCount.onComplete {
      case Success(result) => println(result)
      case Failure(t) => t.printStackTrace
    }
  }
}