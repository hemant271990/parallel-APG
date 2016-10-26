spark-submit --class "org.apache.spark.mllib.optimization.Apg" --master yarn --num-executors 7 --executor-cores 8 --executor-memory 25G target/scala-2.11/accelerated-proximal-gradient_2.11-1.0.jar
