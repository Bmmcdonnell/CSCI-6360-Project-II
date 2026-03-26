error id: file://<HOME>/Library/Mobile%20Documents/com~apple~CloudDocs/CSCI-6360-Project-II/Project_2_2L.scala:scalation.
file://<HOME>/Library/Mobile%20Documents/com~apple~CloudDocs/CSCI-6360-Project-II/Project_2_2L.scala
empty definition using pc, found symbol in pc: 
empty definition using semanticdb
empty definition using fallback
non-local guesses:
	 -scalation/mathstat/scalation.
	 -scalation/modeling/scalation.
	 -ActivationFun.scalation.
	 -scalation/mathstat/Scala2LaTeX.scalation.
	 -scalation.
	 -scala/Predef.scalation.
offset: 96
uri: file://<HOME>/Library/Mobile%20Documents/com~apple~CloudDocs/CSCI-6360-Project-II/Project_2_2L.scala
text:
```scala
package scalation
package modeling
package neuralnet

import scalation.mathstat._
import scalati@@on.modeling._
import ActivationFun._
import scalation.mathstat.Scala2LaTeX._
import scalation.scala2d.savePlot
// import java.awt.Window

@main def P2AutoMPG2L (): Unit =
    // val oxFname = Array ("intercept", "displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin_2", "origin_3")
    val xFname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin_2", "origin_3")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_auto_mpg_with_intercept.csv", 1, sp=',')  // Load the dataset, skipping the header row
    val ox = oxy.not(?, 9)                                                       // Get the first 9 columns as the feature matrix
    val x = ox.not(?, 0)                                                         // Remove the intercept
    val y = oxy(?, 9)                                                            // Get the 10th column as the response vector
    val yy = MatrixD.fromVector (y)                                              // Turn the m-vector y into an m-by-1 matrix

    // --- Train-Test Split (80-20) ---
    val permGen = scalation.mathstat.TnT_Split.makePermGen (ox.dim)              // Make a permutation generator
    val nTest = (ox.dim * 0.2).toInt                                             // 80% training, 20% testing
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, nTest)           // Get test indices for 80-20 split

    // val (oxTest, oxTrain) = TnT_Split (ox, idx)                                  // TnT split the dataset ox (row split)
    val (xTest, xTrain) = TnT_Split (x, idx)                                     // TnT split the dataset x (row split)
    val (yyTest, yyTrain) = TnT_Split (yy, idx)                                  // TnT split the response vector y (row split)
    // val yTrain = yyTrain.col(0)                                                  // Get the train response vector from the test response matrix
    val yTest = yyTest.col(0)                                                    // Get the test response vector from the test response matrix

    val rQ: Range = 0 until 15

    Optimizer.hp("eta") = 0.001                                  // set the learning rate
    Optimizer.hp("bSize") = 1                                  // set the batch size

    banner("In-Sample")
    var mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)
    val (ypIS_, qof1Full) = mod.trainNtest()()                                    // Train and test on the full dataset
    val ypIS = ypIS_.col(0)
    val qof1_ = qof1Full(rQ)                                                      // Extract core QoF metrics
    val qof1 = qof1_.col(0)
    val (yOrd, ypOrdIS) = orderByY(y, ypIS)                                      // Order by actual response for visualization
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${mod.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameIS = s"Auto_MPG_P2/Scalation_2L_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot)                                    // Save high-resolution plot
    // println(mod.summary())                                                       // Print parameter/coefficient statistics
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)                                         // Re-instantiate NN
    val (ypOOS_, qof2Full) = mod.trainNtest(xTrain, yyTrain)(xTest, yyTest)       // Train on training data, test on validation data
    val ypOOS = ypOOS_.col(0)
    val qof2_ = qof2Full(rQ)                                                      // Extract core QoF metrics
    val qof2 = qof2_.col(0)
    val (yTestOrd, ypOrdOOS) = orderByY(yTest, ypOOS)                            // Order by actual test response for visualization
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${mod.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameOOS = s"Auto_MPG_P2/Scalation_2L_80_20.png"
    savePlot(fileNameOOS, plot8020)                                       // Save high-resolution plot
    // println(reg.summary())                                                       // Print parameter/coefficient statistics

    // // --- 5-Fold Cross-Validation ---
    // banner("5-Fold CV")
    // banner("Cross-Validation")
    // mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)                                         // Re-instantiate for CV

    // val stats = mod.crossValidate()

    val nQ = 15
    val rowName = modeling.qoF_names.take(nQ)

    val ColName = "Metric, In-Sample, 80-20 Split"
    val Caption = s"Scalation - Auto MPG 2L NN"
    val Name    = s"Scalation - Auto MPG 2L NN"
    val Qofs    = MatrixD (qof1, qof2).transpose             // Create metrics for both point and interval predictions
    val Latex   = make_doc (make_table (Caption, Name, Qofs, ColName, rowName))
    println (Latex)

    // FitM.showQofStatTable (stats)

    banner("Finished")

    System.exit(0)
end P2AutoMPG2L


@main def P2Housing2L (): Unit =
    // val oxFname = Array("intercept", "longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity_INLAND", "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN")
    val xFname = Array ("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity_INLAND", "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_housing_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 13)                                      
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 13)                                            
    val yy = MatrixD.fromVector (y)                                

    // --- Train-Test Split (80-20) ---
    val permGen = scalation.mathstat.TnT_Split.makePermGen (ox.dim)              
    val nTest = (ox.dim * 0.2).toInt                                             
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, nTest)           

    // val (oxTest, oxTrain) = TnT_Split (ox, idx)                                  
    val (xTest, xTrain) = TnT_Split (x, idx)                                 
    val (yyTest, yyTrain) = TnT_Split (yy, idx)                                  
    // val yTrain = yyTrain.col(0)                                                      
    val yTest = yyTest.col(0)                                                        

    val rQ: Range = 0 until 15

    // Optimizer.hp("eta") = 10 // .606                                  // set the learning rate
    // Optimizer.hp("eta") = 20 // 0.632
    // Optimizer.hp("eta") = 30 // 0.636
    Optimizer.hp("eta") = 25 //0.637
    Optimizer.hp("bSize") = 32                                  // set the batch size

    banner("In-Sample")
    var mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)
    val (ypIS_, qof1Full) = mod.trainNtest()()                                    // Train and test on the full dataset
    val ypIS = ypIS_.col(0)
    val qof1_ = qof1Full(rQ)                                                      // Extract core QoF metrics
    val qof1 = qof1_.col(0)
    val (yOrd, ypOrdIS) = orderByY(y, ypIS)                                      // Order by actual response for visualization
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${mod.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameIS = s"Housing_P2/Scalation_2L_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot)                                    // Save high-resolution plot
    // println(mod.summary())                                                       // Print parameter/coefficient statistics
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)                                         // Re-instantiate NN
    val (ypOOS_, qof2Full) = mod.trainNtest(xTrain, yyTrain)(xTest, yyTest)       // Train on training data, test on validation data
    val ypOOS = ypOOS_.col(0)
    val qof2_ = qof2Full(rQ)                                                      // Extract core QoF metrics
    val qof2 = qof2_.col(0)
    val (yTestOrd, ypOrdOOS) = orderByY(yTest, ypOOS)                            // Order by actual test response for visualization
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${mod.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameOOS = s"Housing_P2/Scalation_2L_80_20.png"
    savePlot(fileNameOOS, plot8020)                                       // Save high-resolution plot
    // println(reg.summary())                                                       // Print parameter/coefficient statistics

    // // --- 5-Fold Cross-Validation ---
    // banner("5-Fold CV")
    // banner("Cross-Validation")
    // mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)                                         // Re-instantiate for CV

    // val stats = mod.crossValidate()

    val nQ = 15
    val rowName = modeling.qoF_names.take(nQ)

    val ColName = "Metric, In-Sample, 80-20 Split"
    val Caption = s"Scalation - Auto MPG 2L NN"
    val Name    = s"Scalation - Auto MPG 2L NN"
    val Qofs    = MatrixD (qof1, qof2).transpose             // Create metrics for both point and interval predictions
    val Latex   = make_doc (make_table (Caption, Name, Qofs, ColName, rowName))
    println (Latex)

    // FitM.showQofStatTable (stats)

    banner("Finished")

    System.exit(0)
end P2Housing2L


@main def P2Insurance2L (): Unit =
    // val oxFname = Array("intercept", "age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")
    val xFname = Array ("age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_insurance_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 9)                                       
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 9)                                             
    val yy = MatrixD.fromVector (y)                                

    // --- Train-Test Split (80-20) ---
    val permGen = scalation.mathstat.TnT_Split.makePermGen (ox.dim)              
    val nTest = (ox.dim * 0.2).toInt                                             
    val idx = scalation.mathstat.TnT_Split.testIndices(permGen, nTest)           

    // val (oxTest, oxTrain) = TnT_Split (ox, idx)                                  
    val (xTest, xTrain) = TnT_Split (x, idx)                                 
    val (yyTest, yyTrain) = TnT_Split (yy, idx)                                  
    // val yTrain = yyTrain.col(0)                                                      
    val yTest = yyTest.col(0)                                                        
    
    val rQ: Range = 0 until 15

    // Optimizer.hp("eta") = 10 // 0.326                                  // set the learning rate
    // Optimizer.hp("eta") = 50 // 0.657
    // Optimizer.hp("eta") = 100 // 0.705
    // Optimizer.hp("eta") = 150 // 0.716
    Optimizer.hp("eta") = 200 // 0.717
    Optimizer.hp("bSize") = 32                                  // set the batch size

    banner("In-Sample")
    var mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)
    val (ypIS_, qof1Full) = mod.trainNtest()()                                    // Train and test on the full dataset
    val ypIS = ypIS_.col(0)
    val qof1_ = qof1Full(rQ)                                                      // Extract core QoF metrics
    val qof1 = qof1_.col(0)
    val (yOrd, ypOrdIS) = orderByY(y, ypIS)                                      // Order by actual response for visualization
    val inSamplePlot = new Plot(null, yOrd, ypOrdIS, s"Plot ${mod.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameIS = s"Insurance_P2/Scalation_2L_In_Sample.png"
    savePlot(fileNameIS, inSamplePlot)                                    // Save high-resolution plot
    // println(mod.summary())                                                       // Print parameter/coefficient statistics
    
    // --- 80-20 Split Evaluation ---
    banner("80-20 Split")
    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)                                         // Re-instantiate NN
    val (ypOOS_, qof2Full) = mod.trainNtest(xTrain, yyTrain)(xTest, yyTest)       // Train on training data, test on validation data
    val ypOOS = ypOOS_.col(0)
    val qof2_ = qof2Full(rQ)                                                      // Extract core QoF metrics
    val qof2 = qof2_.col(0)
    val (yTestOrd, ypOrdOOS) = orderByY(yTest, ypOOS)                            // Order by actual test response for visualization
    val plot8020 = new Plot(null, yTestOrd, ypOrdOOS, s"Plot ${mod.modelName} predictions: yy black/actual vs. yp red/predicted", lines = true)
    Thread.sleep(1000)                                                           // Pause to allow Java Swing to render
    val fileNameOOS = s"Insurance_P2/Scalation_2L_80_20.png"
    savePlot(fileNameOOS, plot8020)                                       // Save high-resolution plot
    // println(reg.summary())                                                       // Print parameter/coefficient statistics

    // // // --- 5-Fold Cross-Validation ---
    // banner("5-Fold CV")
    // banner("Cross-Validation")
    // mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)                                         // Re-instantiate for CV

    // val stats = mod.crossValidate()

    val nQ = 15
    val rowName = modeling.qoF_names.take(nQ)

    val ColName = "Metric, In-Sample, 80-20 Split"
    val Caption = s"Scalation - Auto MPG 2L NN"
    val Name    = s"Scalation - Auto MPG 2L NN"
    val Qofs    = MatrixD (qof1, qof2).transpose             // Create metrics for both point and interval predictions
    val Latex   = make_doc (make_table (Caption, Name, Qofs, ColName, rowName))
    println (Latex)

    // FitM.showQofStatTable (stats)
    
    banner("Finished")

    System.exit(0)
end P2Insurance2L
```


#### Short summary: 

empty definition using pc, found symbol in pc: 