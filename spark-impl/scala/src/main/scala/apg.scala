import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.optimization.{LeastSquaresGradient, L1Updater, SimpleUpdater}
import scala.util
import scala.math.sqrt
import scala.io._
import breeze.linalg.{norm, DenseVector => BDV}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.rdd.RDD

object APG {

	def main(args: Array[String]) {

		val conf = new SparkConf().setAppName("APG")
	    val sc = new SparkContext(conf)
		val MAX_ITER = 100
		val d = 4
		val N = 1000
		// more details: https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/util/LinearDataGenerator.scala
		// 			   : http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.util.LinearDataGenerator$
		val data = LinearDataGenerator.generateLinearRDD(sc, N, d, 1 )
		//data.collect().foreach(println)

		var initialWeights = Vectors.zeros(d)
		var weights = Vectors.dense(initialWeights.toArray)
		val n = weights.size
		var t = 1
		val miniBatchFraction = 1.0
		val gradient = new LeastSquaresGradient()
		val simpleUpdater = new SimpleUpdater()
		val l1Updater = new L1Updater()
		var previousWeights: Vector = weights
	    var currentWeights: Vector = weights
		var stepSize = 0.005
		var regParam = 1
		var currentZ = Vectors.zeros(d)
		var gama_prev : Double = 1
		var gama_curr : Double = 1
		var gama_next : Double = 1
		var u_curr: Vector = weights
		var u_next: Vector = weights

		while(t <= MAX_ITER) {

			// 1. compute Lasso gradient: http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.optimization.LeastSquaresGradient

			val bcWeights = data.context.broadcast(u_curr)

			// Sample a subset (fraction miniBatchFraction) of the total data
			// compute and sum up the subgradients on this subset (this is one map-reduce)

			val (gradientSum, lossSum) = data.sample(false, miniBatchFraction, 42 + t)
			 .treeAggregate(zeroValue = (Vectors.zeros(d), 0.0))(
    	      seqOp = (c, v) => {
    	        // c: (grad, loss), v: (label, features)
				val loss = gradient.compute(v.features, v.label, bcWeights.value, c._1)
	            (c._1, c._2 + loss)
    	      },
    	      combOp = (c1, c2) => {
    	        // c: (grad, loss)
				var i = 0
				var sum = new Array[Double](c1._1.size)
        		while(i < c1._1.size){
            		sum(i) = c1._1(i)+c2._1(i)
		            i += 1 
        		}
    	        (Vectors.dense(sum), c1._2 + c2._2)
    	      })

			//println("Gradient: " + gradientSum)
	

			// 2. update least square using simple updater: http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.optimization.SimpleUpdater

			val simpleUpdate = simpleUpdater.compute(u_curr, new DenseVector(gradientSum.toArray), stepSize, t, regParam)
			currentZ = simpleUpdate._1

			
			// 3. L1 regularizer updates: http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.optimization.L1Updater

			val L1RegUpdate = l1Updater.compute(currentZ, Vectors.zeros(d), stepSize, 1, regParam)
			currentWeights = L1RegUpdate._1

			gama_next = (1 + sqrt(1 + 4*gama_curr*gama_curr))/2.0
			var i = 0
			var u_tmp = new Array[Double](d)
			while(i < d) {
				u_tmp(i) = currentWeights(i) + (gama_prev/gama_next)*(currentWeights(i) - previousWeights(i))
				i += 1
			}
			u_next = Vectors.dense(u_tmp)

			previousWeights = currentWeights
			gama_prev = gama_curr
			gama_curr = gama_next
			u_curr = u_next
			
			t += 1
			println("Current weights: " +currentWeights)

		}
	}
}
