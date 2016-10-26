//package org.apache.spark.mllib.regression

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.optimization.{LeastSquaresGradient, L1Updater, SimpleUpdater}
import scala.util
import breeze.linalg.{norm, DenseVector => BDV}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.rdd.RDD

object PG {

	def main(args: Array[String]) {

		val conf = new SparkConf().setAppName("APG")
	    val sc = new SparkContext(conf)
		val MAX_ITER = 100
		val d = 4
		val N = 1000
		// more details: https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/util/LinearDataGenerator.scala
		// 			   : http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.util.LinearDataGenerator$
		val data = LinearDataGenerator.generateLinearRDD(sc, N, d, 1 )
		data.collect().foreach(println)

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
		var stepSize = 0.05
		var regParam = 0.01
		var currentZ = Vectors.zeros(d)

		while(t <= MAX_ITER) {

			// 1. compute Lasso gradient: http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.optimization.LeastSquaresGradient

			//val bcWeights = data.context.broadcast(weights)
			val bcWeights = sc.broadcast(weights)

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

			println("Gradient: " + gradientSum)
	

			// 2. update least square using simple updater: http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.optimization.SimpleUpdater

			val simpleUpdate = simpleUpdater.compute(currentWeights, new DenseVector(gradientSum.toArray), stepSize, t, regParam)
			currentZ = simpleUpdate._1

			
			// 3. L1 regularizer updates: http://spark.apache.org/docs/1.3.0/api/scala/index.html#org.apache.spark.mllib.optimization.L1Updater

			val L1RegUpdate = l1Updater.compute(currentZ, Vectors.zeros(weights.size), stepSize, 1, regParam)
			previousWeights = currentWeights
			currentWeights = L1RegUpdate._1
			
			t += 1
			println("Current weights: " +currentWeights)
		}

		
	}
}
