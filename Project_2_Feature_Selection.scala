package scalation
package modeling
package neuralnet

import scalation.mathstat._
import scalation.modeling._
import ActivationFun._
import scalation.scala2d.savePlot
import java.awt.Window


@main def P2AutoMPG2LFeatureSelection (): Unit =

    val xFname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin_2", "origin_3")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_auto_mpg_with_intercept.csv", 1, sp=',')  // Load the dataset, skipping the header row
    val ox = oxy.not(?, 9)                                                       // Get the first 9 columns as the feature matrix
    val x = ox.not(?, 0)                                                         // Remove the intercept
    val y = oxy(?, 9)                                                            // Get the 10th column as the response vector
    val yy = MatrixD.fromVector (y)                                              // Turn the m-vector y into an m-by-1 matrix

    Optimizer.hp("eta") = 0.001                                  // set the learning rate
    Optimizer.hp("bSize") = 1                                  // set the batch size

    
    // ==========================================
    // --- 2 Layer Neural Network ---
    // ==========================================
    banner(s"Auto MPG 2L NN")
    var mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (forCols, forRSq) = mod.forwardSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val forAicList = new VectorD(forCols.size)
    val forBicList = new VectorD(forCols.size)

    for k <- 0 until forCols.size do
        val forSubCols = forCols.slice(0, k + 1)                                   // Subset top k+1 features
        val forxSub = x(?, forSubCols)

        val subNN= new NeuralNet_2L (forxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, forQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        forAicList(k) = forQof.col(0)(13)                                                 // Store AIC
        forBicList(k) = forQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val forColsArray = forCols.toArray
    val forColsArrayD = forColsArray.map(_.toDouble)
    val forColsVecD = VectorD(forColsArrayD)                                       // Convert for display
    val forAicBicMatrix = MatrixD(forColsVecD, forAicList, forBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $forAicBicMatrix")

    // Generate and save plots
    if forCols.nonEmpty && forRSq.dim > 0 && forRSq.dim2 > 0 then
        val forRSqPlot = new PlotM(null, forRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Forward Selection on Auto MPG", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val forRSqFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_2L_rSq_Forward.png"
        savePlot(forRSqFileName, forRSqPlot, 2.1)
        
        val forAicBicPlot = new PlotM(null, MatrixD(forAicList, forBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Forward Selection on Auto MPG", lines = true)
        Thread.sleep(1000)
        val forAicBicFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_2L_AIC_BIC_Forward.png"
        savePlot(forAicBicFileName, forAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val forColsList = forColsArray.toList
    var forColNames = List.fill(forCols.size)("")
    for i <- 0 until forCols.size do
        forColNames = forColNames.updated(i, xFname(forColsList(i)))
    end for


    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (backCols, backRSq) = mod.backwardElimAll()

    // Calculate AIC and BIC for each stage of the selection process
    val backAicList = new VectorD(backCols.size)
    val backBicList = new VectorD(backCols.size)

    for k <- 0 until backCols.size do
        val backSubCols = backCols.slice(0, k + 1)                                   // Subset top k+1 features
        val backxSub = x(?, backSubCols)

        val subNN= new NeuralNet_2L (backxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, backQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        backAicList(k) = backQof.col(0)(13)                                                 // Store AIC
        backBicList(k) = backQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val backColsArray = backCols.toArray
    val backColsArrayD = backColsArray.map(_.toDouble)
    val backColsVecD = VectorD(backColsArrayD)                                       // Convert for display
    val backAicBicMatrix = MatrixD(backColsVecD, backAicList, backBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $backAicBicMatrix")

    // Generate and save plots
    if backCols.nonEmpty && backRSq.dim > 0 && backRSq.dim2 > 0 then
        val backRSqPlot = new PlotM(null, backRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Backward Elimination on Auto MPG", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val backRSqFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_2L_rSq_Backward.png"
        savePlot(backRSqFileName, backRSqPlot, 2.1)
        
        val backAicBicPlot = new PlotM(null, MatrixD(backAicList, backBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Backward Elimination on Auto MPG", lines = true)
        Thread.sleep(1000)
        val backAicBicFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_2L_AIC_BIC_Backward.png"
        savePlot(backAicBicFileName, backAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Backward Elimination: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val backColsList = backColsArray.toList
    var backColNames = List.fill(backCols.size)("")
    for i <- 0 until backCols.size do
        backColNames = backColNames.updated(i, xFname(backColsList(i)))
    end for

    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (stepCols, stepRSq) = mod.stepwiseSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val stepAicList = new VectorD(stepCols.size)
    val stepBicList = new VectorD(stepCols.size)

    for k <- 0 until stepCols.size do
        val stepSubCols = stepCols.slice(0, k + 1)                                   // Subset top k+1 features
        val stepxSub = x(?, stepSubCols)

        val subNN= new NeuralNet_2L (stepxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, stepQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        stepAicList(k) = stepQof.col(0)(13)                                                 // Store AIC
        stepBicList(k) = stepQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val stepColsArray = stepCols.toArray
    val stepColsArrayD = stepColsArray.map(_.toDouble)
    val stepColsVecD = VectorD(stepColsArrayD)                                       // Convert for display
    val stepAicBicMatrix = MatrixD(stepColsVecD, stepAicList, stepBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $stepAicBicMatrix")

    // Generate and save plots
    if stepCols.nonEmpty && stepRSq.dim > 0 && stepRSq.dim2 > 0 then
        val stepRSqPlot = new PlotM(null, stepRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Stepwise Selection on Auto MPG", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val stepRSqFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_2L_rSq_Stepwise.png"
        savePlot(stepRSqFileName, stepRSqPlot, 2.1)
        
        val stepAicBicPlot = new PlotM(null, MatrixD(stepAicList, stepBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Stepwise Selection on Auto MPG", lines = true)
        Thread.sleep(1000)
        val stepAicBicFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_2L_AIC_BIC_Stepwise.png"
        savePlot(stepAicBicFileName, stepAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val stepColsList = stepColsArray.toList
    var stepColNames = List.fill(stepCols.size)("")
    for i <- 0 until stepCols.size do
        stepColNames = stepColNames.updated(i, xFname(stepColsList(i)))
    end for

    println (s"Scalation Forward Selection Order: $forColNames")
    println (s"Scalation Backward Elimination Reversed Order: $backColNames")
    println (s"Scalation Stepwise Selection Order: $stepColNames")

end P2AutoMPG2LFeatureSelection


@main def P2Housing2LFeatureSelection (): Unit =

    val xFname = Array ("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity_INLAND", "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_housing_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 13)                                      
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 13)                                            
    val yy = MatrixD.fromVector (y)

    Optimizer.hp("eta") = 25                                  // set the learning rate
    Optimizer.hp("bSize") = 32                                  // set the batch size

    
    // ==========================================
    // --- 2 Layer Neural Network ---
    // ==========================================
    banner(s"California House Prices 2L NN")
    var mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (forCols, forRSq) = mod.forwardSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val forAicList = new VectorD(forCols.size)
    val forBicList = new VectorD(forCols.size)

    for k <- 0 until forCols.size do
        val forSubCols = forCols.slice(0, k + 1)                                   // Subset top k+1 features
        val forxSub = x(?, forSubCols)

        val subNN= new NeuralNet_2L (forxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, forQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        forAicList(k) = forQof.col(0)(13)                                                 // Store AIC
        forBicList(k) = forQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val forColsArray = forCols.toArray
    val forColsArrayD = forColsArray.map(_.toDouble)
    val forColsVecD = VectorD(forColsArrayD)                                       // Convert for display
    val forAicBicMatrix = MatrixD(forColsVecD, forAicList, forBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $forAicBicMatrix")

    // Generate and save plots
    if forCols.nonEmpty && forRSq.dim > 0 && forRSq.dim2 > 0 then
        val forRSqPlot = new PlotM(null, forRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Forward Selection on California House Prices", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val forRSqFileName = s"Housing_P2_Feature_Selection/Scalation_2L_rSq_Forward.png"
        savePlot(forRSqFileName, forRSqPlot, 2.1)
        
        val forAicBicPlot = new PlotM(null, MatrixD(forAicList, forBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Forward Selection on California House Prices", lines = true)
        Thread.sleep(1000)
        val forAicBicFileName = s"Housing_P2_Feature_Selection/Scalation_2L_AIC_BIC_Forward.png"
        savePlot(forAicBicFileName, forAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val forColsList = forColsArray.toList
    var forColNames = List.fill(forCols.size)("")
    for i <- 0 until forCols.size do
        forColNames = forColNames.updated(i, xFname(forColsList(i)))
    end for


    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (backCols, backRSq) = mod.backwardElimAll()

    // Calculate AIC and BIC for each stage of the selection process
    val backAicList = new VectorD(backCols.size)
    val backBicList = new VectorD(backCols.size)

    for k <- 0 until backCols.size do
        val backSubCols = backCols.slice(0, k + 1)                                   // Subset top k+1 features
        val backxSub = x(?, backSubCols)

        val subNN= new NeuralNet_2L (backxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, backQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        backAicList(k) = backQof.col(0)(13)                                                 // Store AIC
        backBicList(k) = backQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val backColsArray = backCols.toArray
    val backColsArrayD = backColsArray.map(_.toDouble)
    val backColsVecD = VectorD(backColsArrayD)                                       // Convert for display
    val backAicBicMatrix = MatrixD(backColsVecD, backAicList, backBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $backAicBicMatrix")

    // Generate and save plots
    if backCols.nonEmpty && backRSq.dim > 0 && backRSq.dim2 > 0 then
        val backRSqPlot = new PlotM(null, backRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Backward Elimination on California House Prices", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val backRSqFileName = s"Housing_P2_Feature_Selection/Scalation_2L_rSq_Backward.png"
        savePlot(backRSqFileName, backRSqPlot, 2.1)
        
        val backAicBicPlot = new PlotM(null, MatrixD(backAicList, backBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Backward Elimination on California House Prices", lines = true)
        Thread.sleep(1000)
        val backAicBicFileName = s"Housing_P2_Feature_Selection/Scalation_2L_AIC_BIC_Backward.png"
        savePlot(backAicBicFileName, backAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Backward Elimination: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val backColsList = backColsArray.toList
    var backColNames = List.fill(backCols.size)("")
    for i <- 0 until backCols.size do
        backColNames = backColNames.updated(i, xFname(backColsList(i)))
    end for

    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (stepCols, stepRSq) = mod.stepwiseSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val stepAicList = new VectorD(stepCols.size)
    val stepBicList = new VectorD(stepCols.size)

    for k <- 0 until stepCols.size do
        val stepSubCols = stepCols.slice(0, k + 1)                                   // Subset top k+1 features
        val stepxSub = x(?, stepSubCols)

        val subNN= new NeuralNet_2L (stepxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, stepQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        stepAicList(k) = stepQof.col(0)(13)                                                 // Store AIC
        stepBicList(k) = stepQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val stepColsArray = stepCols.toArray
    val stepColsArrayD = stepColsArray.map(_.toDouble)
    val stepColsVecD = VectorD(stepColsArrayD)                                       // Convert for display
    val stepAicBicMatrix = MatrixD(stepColsVecD, stepAicList, stepBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $stepAicBicMatrix")

    // Generate and save plots
    if stepCols.nonEmpty && stepRSq.dim > 0 && stepRSq.dim2 > 0 then
        val stepRSqPlot = new PlotM(null, stepRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Stepwise Selection on California House Prices", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val stepRSqFileName = s"Housing_P2_Feature_Selection/Scalation_2L_rSq_Stepwise.png"
        savePlot(stepRSqFileName, stepRSqPlot, 2.1)
        
        val stepAicBicPlot = new PlotM(null, MatrixD(stepAicList, stepBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Stepwise Selection on California House Prices", lines = true)
        Thread.sleep(1000)
        val stepAicBicFileName = s"Housing_P2_Feature_Selection/Scalation_2L_AIC_BIC_Stepwise.png"
        savePlot(stepAicBicFileName, stepAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val stepColsList = stepColsArray.toList
    var stepColNames = List.fill(stepCols.size)("")
    for i <- 0 until stepCols.size do
        stepColNames = stepColNames.updated(i, xFname(stepColsList(i)))
    end for

    println (s"Scalation Forward Selection Order: $forColNames")
    println (s"Scalation Backward Elimination Reversed Order: $backColNames")
    println (s"Scalation Stepwise Selection Order: $stepColNames")

end P2Housing2LFeatureSelection


@main def P2Insurance2LFeatureSelection (): Unit =

    val xFname = Array ("age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_insurance_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 9)                                       
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 9)                                             
    val yy = MatrixD.fromVector (y) 

    Optimizer.hp("eta") = 200                                  // set the learning rate
    Optimizer.hp("bSize") = 32                                  // set the batch size

    
    // ==========================================
    // --- 2 Layer Neural Network ---
    // ==========================================
    banner(s"Insurance Charges 2L NN")
    var mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (forCols, forRSq) = mod.forwardSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val forAicList = new VectorD(forCols.size)
    val forBicList = new VectorD(forCols.size)

    for k <- 0 until forCols.size do
        val forSubCols = forCols.slice(0, k + 1)                                   // Subset top k+1 features
        val forxSub = x(?, forSubCols)

        val subNN= new NeuralNet_2L (forxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, forQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        forAicList(k) = forQof.col(0)(13)                                                 // Store AIC
        forBicList(k) = forQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val forColsArray = forCols.toArray
    val forColsArrayD = forColsArray.map(_.toDouble)
    val forColsVecD = VectorD(forColsArrayD)                                       // Convert for display
    val forAicBicMatrix = MatrixD(forColsVecD, forAicList, forBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $forAicBicMatrix")

    // Generate and save plots
    if forCols.nonEmpty && forRSq.dim > 0 && forRSq.dim2 > 0 then
        val forRSqPlot = new PlotM(null, forRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Forward Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val forRSqFileName = s"Insurance_P2_Feature_Selection/Scalation_2L_rSq_Forward.png"
        savePlot(forRSqFileName, forRSqPlot, 2.1)
        
        val forAicBicPlot = new PlotM(null, MatrixD(forAicList, forBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Forward Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)
        val forAicBicFileName = s"Insurance_P2_Feature_Selection/Scalation_2L_AIC_BIC_Forward.png"
        savePlot(forAicBicFileName, forAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val forColsList = forColsArray.toList
    var forColNames = List.fill(forCols.size)("")
    for i <- 0 until forCols.size do
        forColNames = forColNames.updated(i, xFname(forColsList(i)))
    end for


    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (backCols, backRSq) = mod.backwardElimAll()

    // Calculate AIC and BIC for each stage of the selection process
    val backAicList = new VectorD(backCols.size)
    val backBicList = new VectorD(backCols.size)

    for k <- 0 until backCols.size do
        val backSubCols = backCols.slice(0, k + 1)                                   // Subset top k+1 features
        val backxSub = x(?, backSubCols)

        val subNN= new NeuralNet_2L (backxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, backQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        backAicList(k) = backQof.col(0)(13)                                                 // Store AIC
        backBicList(k) = backQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val backColsArray = backCols.toArray
    val backColsArrayD = backColsArray.map(_.toDouble)
    val backColsVecD = VectorD(backColsArrayD)                                       // Convert for display
    val backAicBicMatrix = MatrixD(backColsVecD, backAicList, backBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $backAicBicMatrix")

    // Generate and save plots
    if backCols.nonEmpty && backRSq.dim > 0 && backRSq.dim2 > 0 then
        val backRSqPlot = new PlotM(null, backRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Backward Elimination on Insurance Charges", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val backRSqFileName = s"Insurance_P2_Feature_Selection/Scalation_2L_rSq_Backward.png"
        savePlot(backRSqFileName, backRSqPlot, 2.1)
        
        val backAicBicPlot = new PlotM(null, MatrixD(backAicList, backBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Backward Elimination on Insurance Charges", lines = true)
        Thread.sleep(1000)
        val backAicBicFileName = s"Insurance_P2_Feature_Selection/Scalation_2L_AIC_BIC_Backward.png"
        savePlot(backAicBicFileName, backAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Backward Elimination: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val backColsList = backColsArray.toList
    var backColNames = List.fill(backCols.size)("")
    for i <- 0 until backCols.size do
        backColNames = backColNames.updated(i, xFname(backColsList(i)))
    end for

    mod = new NeuralNet_2L (x, yy, f = f_reLU, fname_ = xFname)

    val (stepCols, stepRSq) = mod.stepwiseSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val stepAicList = new VectorD(stepCols.size)
    val stepBicList = new VectorD(stepCols.size)

    for k <- 0 until stepCols.size do
        val stepSubCols = stepCols.slice(0, k + 1)                                   // Subset top k+1 features
        val stepxSub = x(?, stepSubCols)

        val subNN= new NeuralNet_2L (stepxSub, yy, f = f_reLU, fname_ = xFname)
        val (_, stepQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        stepAicList(k) = stepQof.col(0)(13)                                                 // Store AIC
        stepBicList(k) = stepQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val stepColsArray = stepCols.toArray
    val stepColsArrayD = stepColsArray.map(_.toDouble)
    val stepColsVecD = VectorD(stepColsArrayD)                                       // Convert for display
    val stepAicBicMatrix = MatrixD(stepColsVecD, stepAicList, stepBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $stepAicBicMatrix")

    // Generate and save plots
    if stepCols.nonEmpty && stepRSq.dim > 0 && stepRSq.dim2 > 0 then
        val stepRSqPlot = new PlotM(null, stepRSq.ᵀ, Regression.metrics, s"R^2 vs n for 2 Layer Neural Network Stepwise Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val stepRSqFileName = s"Insurance_P2_Feature_Selection/Scalation_2L_rSq_Stepwise.png"
        savePlot(stepRSqFileName, stepRSqPlot, 2.1)
        
        val stepAicBicPlot = new PlotM(null, MatrixD(stepAicList, stepBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 2 Layer Neural Network Stepwise Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)
        val stepAicBicFileName = s"Insurance_P2_Feature_Selection/Scalation_2L_AIC_BIC_Stepwise.png"
        savePlot(stepAicBicFileName, stepAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 2L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val stepColsList = stepColsArray.toList
    var stepColNames = List.fill(stepCols.size)("")
    for i <- 0 until stepCols.size do
        stepColNames = stepColNames.updated(i, xFname(stepColsList(i)))
    end for

    println (s"Scalation Forward Selection Order: $forColNames")
    println (s"Scalation Backward Elimination Reversed Order: $backColNames")
    println (s"Scalation Stepwise Selection Order: $stepColNames")

end P2Insurance2LFeatureSelection


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


@main def P2AutoMPG3LFeatureSelection (): Unit =

    val xFname = Array ("displacement", "cylinders", "horsepower", "weight", "acceleration", "modelyear", "origin_2", "origin_3")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_auto_mpg_with_intercept.csv", 1, sp=',')  // Load the dataset, skipping the header row
    val ox = oxy.not(?, 9)                                                       // Get the first 9 columns as the feature matrix
    val x = ox.not(?, 0)                                                         // Remove the intercept
    val y = oxy(?, 9)                                                            // Get the 10th column as the response vector
    val yy = MatrixD.fromVector (y)                                              // Turn the m-vector y into an m-by-1 matrix

    Optimizer.hp("eta") = 0.01                                  // set the learning rate
    Optimizer.hp("bSize") = 8                                  // set the batch size
    val actfHidden = f_sigmoid
    val actfOut = f_reLU
    val nz_ = 3 * x.dim2

    
    // ==========================================
    // --- 2 Layer Neural Network ---
    // ==========================================
    banner(s"Auto MPG 3L NN")
    var mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (forCols, forRSq) = mod.forwardSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val forAicList = new VectorD(forCols.size)
    val forBicList = new VectorD(forCols.size)

    for k <- 0 until forCols.size do
        val forSubCols = forCols.slice(0, k + 1)                                   // Subset top k+1 features
        val forxSub = x(?, forSubCols)

        val subNN= NeuralNet_3L.rescale (forxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, forQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        forAicList(k) = forQof.col(0)(13)                                                 // Store AIC
        forBicList(k) = forQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val forColsArray = forCols.toArray
    val forColsArrayD = forColsArray.map(_.toDouble)
    val forColsVecD = VectorD(forColsArrayD)                                       // Convert for display
    val forAicBicMatrix = MatrixD(forColsVecD, forAicList, forBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $forAicBicMatrix")

    // Generate and save plots
    if forCols.nonEmpty && forRSq.dim > 0 && forRSq.dim2 > 0 then
        val forRSqPlot = new PlotM(null, forRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Forward Selection on Auto MPG", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val forRSqFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_3L_rSq_Forward.png"
        savePlot(forRSqFileName, forRSqPlot, 2.1)
        
        val forAicBicPlot = new PlotM(null, MatrixD(forAicList, forBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Forward Selection on Auto MPG", lines = true)
        Thread.sleep(1000)
        val forAicBicFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_3L_AIC_BIC_Forward.png"
        savePlot(forAicBicFileName, forAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val forColsList = forColsArray.toList
    var forColNames = List.fill(forCols.size)("")
    for i <- 0 until forCols.size do
        forColNames = forColNames.updated(i, xFname(forColsList(i)))
    end for


    mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (backCols, backRSq) = mod.backwardElimAll()

    // Calculate AIC and BIC for each stage of the selection process
    val backAicList = new VectorD(backCols.size)
    val backBicList = new VectorD(backCols.size)

    for k <- 0 until backCols.size do
        val backSubCols = backCols.slice(0, k + 1)                                   // Subset top k+1 features
        val backxSub = x(?, backSubCols)

        val subNN= NeuralNet_3L.rescale (backxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, backQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        backAicList(k) = backQof.col(0)(13)                                                 // Store AIC
        backBicList(k) = backQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val backColsArray = backCols.toArray
    val backColsArrayD = backColsArray.map(_.toDouble)
    val backColsVecD = VectorD(backColsArrayD)                                       // Convert for display
    val backAicBicMatrix = MatrixD(backColsVecD, backAicList, backBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $backAicBicMatrix")

    // Generate and save plots
    if backCols.nonEmpty && backRSq.dim > 0 && backRSq.dim2 > 0 then
        val backRSqPlot = new PlotM(null, backRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Backward Elimination on Auto MPG", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val backRSqFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_3L_rSq_Backward.png"
        savePlot(backRSqFileName, backRSqPlot, 2.1)
        
        val backAicBicPlot = new PlotM(null, MatrixD(backAicList, backBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Backward Elimination on Auto MPG", lines = true)
        Thread.sleep(1000)
        val backAicBicFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_3L_AIC_BIC_Backward.png"
        savePlot(backAicBicFileName, backAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Backward Elimination: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val backColsList = backColsArray.toList
    var backColNames = List.fill(backCols.size)("")
    for i <- 0 until backCols.size do
        backColNames = backColNames.updated(i, xFname(backColsList(i)))
    end for

    mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (stepCols, stepRSq) = mod.stepwiseSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val stepAicList = new VectorD(stepCols.size)
    val stepBicList = new VectorD(stepCols.size)

    for k <- 0 until stepCols.size do
        val stepSubCols = stepCols.slice(0, k + 1)                                   // Subset top k+1 features
        val stepxSub = x(?, stepSubCols)

        val subNN= NeuralNet_3L.rescale (stepxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, stepQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        stepAicList(k) = stepQof.col(0)(13)                                                 // Store AIC
        stepBicList(k) = stepQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val stepColsArray = stepCols.toArray
    val stepColsArrayD = stepColsArray.map(_.toDouble)
    val stepColsVecD = VectorD(stepColsArrayD)                                       // Convert for display
    val stepAicBicMatrix = MatrixD(stepColsVecD, stepAicList, stepBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $stepAicBicMatrix")

    // Generate and save plots
    if stepCols.nonEmpty && stepRSq.dim > 0 && stepRSq.dim2 > 0 then
        val stepRSqPlot = new PlotM(null, stepRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Stepwise Selection on Auto MPG", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val stepRSqFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_3L_rSq_Stepwise.png"
        savePlot(stepRSqFileName, stepRSqPlot, 2.1)
        
        val stepAicBicPlot = new PlotM(null, MatrixD(stepAicList, stepBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Stepwise Selection on Auto MPG", lines = true)
        Thread.sleep(1000)
        val stepAicBicFileName = s"Auto_MPG_P2_Feature_Selection/Scalation_3L_AIC_BIC_Stepwise.png"
        savePlot(stepAicBicFileName, stepAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val stepColsList = stepColsArray.toList
    var stepColNames = List.fill(stepCols.size)("")
    for i <- 0 until stepCols.size do
        stepColNames = stepColNames.updated(i, xFname(stepColsList(i)))
    end for

    println (s"Scalation Forward Selection Order: $forColNames")
    println (s"Scalation Backward Elimination Reversed Order: $backColNames")
    println (s"Scalation Stepwise Selection Order: $stepColNames")

end P2AutoMPG3LFeatureSelection


@main def P2Housing3LFeatureSelection (): Unit =

    val xFname = Array ("longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income", "ocean_proximity_INLAND", "ocean_proximity_ISLAND", "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_housing_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 13)                                      
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 13)                                            
    val yy = MatrixD.fromVector (y)

    Optimizer.hp("eta") = 0.01                                  // set the learning rate
    Optimizer.hp("bSize") = 32                                  // set the batch size
    val actfHidden = f_lreLU
    val actfOut = f_id
    val nz_ = 3 * x.dim2

    
    // ==========================================
    // --- 2 Layer Neural Network ---
    // ==========================================
    banner(s"California House Prices 3L NN")
    var mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (forCols, forRSq) = mod.forwardSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val forAicList = new VectorD(forCols.size)
    val forBicList = new VectorD(forCols.size)

    for k <- 0 until forCols.size do
        val forSubCols = forCols.slice(0, k + 1)                                   // Subset top k+1 features
        val forxSub = x(?, forSubCols)

        val subNN= NeuralNet_3L.rescale (forxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, forQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        forAicList(k) = forQof.col(0)(13)                                                 // Store AIC
        forBicList(k) = forQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val forColsArray = forCols.toArray
    val forColsArrayD = forColsArray.map(_.toDouble)
    val forColsVecD = VectorD(forColsArrayD)                                       // Convert for display
    val forAicBicMatrix = MatrixD(forColsVecD, forAicList, forBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $forAicBicMatrix")

    // Generate and save plots
    if forCols.nonEmpty && forRSq.dim > 0 && forRSq.dim2 > 0 then
        val forRSqPlot = new PlotM(null, forRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Forward Selection on California House Prices", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val forRSqFileName = s"Housing_P2_Feature_Selection/Scalation_3L_rSq_Forward.png"
        savePlot(forRSqFileName, forRSqPlot, 2.1)
        
        val forAicBicPlot = new PlotM(null, MatrixD(forAicList, forBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Forward Selection on California House Prices", lines = true)
        Thread.sleep(1000)
        val forAicBicFileName = s"Housing_P2_Feature_Selection/Scalation_3L_AIC_BIC_Forward.png"
        savePlot(forAicBicFileName, forAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val forColsList = forColsArray.toList
    var forColNames = List.fill(forCols.size)("")
    for i <- 0 until forCols.size do
        forColNames = forColNames.updated(i, xFname(forColsList(i)))
    end for


    mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (backCols, backRSq) = mod.backwardElimAll()

    // Calculate AIC and BIC for each stage of the selection process
    val backAicList = new VectorD(backCols.size)
    val backBicList = new VectorD(backCols.size)

    for k <- 0 until backCols.size do
        val backSubCols = backCols.slice(0, k + 1)                                   // Subset top k+1 features
        val backxSub = x(?, backSubCols)

        val subNN= NeuralNet_3L.rescale (backxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, backQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        backAicList(k) = backQof.col(0)(13)                                                 // Store AIC
        backBicList(k) = backQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val backColsArray = backCols.toArray
    val backColsArrayD = backColsArray.map(_.toDouble)
    val backColsVecD = VectorD(backColsArrayD)                                       // Convert for display
    val backAicBicMatrix = MatrixD(backColsVecD, backAicList, backBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $backAicBicMatrix")

    // Generate and save plots
    if backCols.nonEmpty && backRSq.dim > 0 && backRSq.dim2 > 0 then
        val backRSqPlot = new PlotM(null, backRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Backward Elimination on California House Prices", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val backRSqFileName = s"Housing_P2_Feature_Selection/Scalation_3L_rSq_Backward.png"
        savePlot(backRSqFileName, backRSqPlot, 2.1)
        
        val backAicBicPlot = new PlotM(null, MatrixD(backAicList, backBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Backward Elimination on California House Prices", lines = true)
        Thread.sleep(1000)
        val backAicBicFileName = s"Housing_P2_Feature_Selection/Scalation_3L_AIC_BIC_Backward.png"
        savePlot(backAicBicFileName, backAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Backward Elimination: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val backColsList = backColsArray.toList
    var backColNames = List.fill(backCols.size)("")
    for i <- 0 until backCols.size do
        backColNames = backColNames.updated(i, xFname(backColsList(i)))
    end for

    mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (stepCols, stepRSq) = mod.stepwiseSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val stepAicList = new VectorD(stepCols.size)
    val stepBicList = new VectorD(stepCols.size)

    for k <- 0 until stepCols.size do
        val stepSubCols = stepCols.slice(0, k + 1)                                   // Subset top k+1 features
        val stepxSub = x(?, stepSubCols)

        val subNN= NeuralNet_3L.rescale (stepxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, stepQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        stepAicList(k) = stepQof.col(0)(13)                                                 // Store AIC
        stepBicList(k) = stepQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val stepColsArray = stepCols.toArray
    val stepColsArrayD = stepColsArray.map(_.toDouble)
    val stepColsVecD = VectorD(stepColsArrayD)                                       // Convert for display
    val stepAicBicMatrix = MatrixD(stepColsVecD, stepAicList, stepBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $stepAicBicMatrix")

    // Generate and save plots
    if stepCols.nonEmpty && stepRSq.dim > 0 && stepRSq.dim2 > 0 then
        val stepRSqPlot = new PlotM(null, stepRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Stepwise Selection on California House Prices", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val stepRSqFileName = s"Housing_P2_Feature_Selection/Scalation_3L_rSq_Stepwise.png"
        savePlot(stepRSqFileName, stepRSqPlot, 2.1)
        
        val stepAicBicPlot = new PlotM(null, MatrixD(stepAicList, stepBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Stepwise Selection on California House Prices", lines = true)
        Thread.sleep(1000)
        val stepAicBicFileName = s"Housing_P2_Feature_Selection/Scalation_3L_AIC_BIC_Stepwise.png"
        savePlot(stepAicBicFileName, stepAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val stepColsList = stepColsArray.toList
    var stepColNames = List.fill(stepCols.size)("")
    for i <- 0 until stepCols.size do
        stepColNames = stepColNames.updated(i, xFname(stepColsList(i)))
    end for

    println (s"Scalation Forward Selection Order: $forColNames")
    println (s"Scalation Backward Elimination Reversed Order: $backColNames")
    println (s"Scalation Stepwise Selection Order: $stepColNames")

end P2Housing3LFeatureSelection


@main def P2Insurance3LFeatureSelection (): Unit =

    val xFname = Array ("age", "bmi", "children", "sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest")

    // --- Data Loading ---
    val oxy = MatrixD.load ("cleaned_insurance_with_intercept.csv", 1, sp=',')      
    val ox = oxy.not(?, 9)                                       
    val x = ox.not(?, 0)                                         
    val y = oxy(?, 9)                                             
    val yy = MatrixD.fromVector (y) 

    Optimizer.hp("eta") = 0.01                                  // set the learning rate
    Optimizer.hp("bSize") = 16                                  // set the batch size
    val actfHidden = f_lreLU
    val actfOut = f_id
    val nz_ = 2 * x.dim2

    
    // ==========================================
    // --- 2 Layer Neural Network ---
    // ==========================================
    banner(s"Insurance Charges 3L NN")
    var mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (forCols, forRSq) = mod.forwardSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val forAicList = new VectorD(forCols.size)
    val forBicList = new VectorD(forCols.size)

    for k <- 0 until forCols.size do
        val forSubCols = forCols.slice(0, k + 1)                                   // Subset top k+1 features
        val forxSub = x(?, forSubCols)

        val subNN= NeuralNet_3L.rescale (forxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, forQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        forAicList(k) = forQof.col(0)(13)                                                 // Store AIC
        forBicList(k) = forQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val forColsArray = forCols.toArray
    val forColsArrayD = forColsArray.map(_.toDouble)
    val forColsVecD = VectorD(forColsArrayD)                                       // Convert for display
    val forAicBicMatrix = MatrixD(forColsVecD, forAicList, forBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $forAicBicMatrix")

    // Generate and save plots
    if forCols.nonEmpty && forRSq.dim > 0 && forRSq.dim2 > 0 then
        val forRSqPlot = new PlotM(null, forRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Forward Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val forRSqFileName = s"Insurance_P2_Feature_Selection/Scalation_3L_rSq_Forward.png"
        savePlot(forRSqFileName, forRSqPlot, 2.1)
        
        val forAicBicPlot = new PlotM(null, MatrixD(forAicList, forBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Forward Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)
        val forAicBicFileName = s"Insurance_P2_Feature_Selection/Scalation_3L_AIC_BIC_Forward.png"
        savePlot(forAicBicFileName, forAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val forColsList = forColsArray.toList
    var forColNames = List.fill(forCols.size)("")
    for i <- 0 until forCols.size do
        forColNames = forColNames.updated(i, xFname(forColsList(i)))
    end for


    mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (backCols, backRSq) = mod.backwardElimAll()

    // Calculate AIC and BIC for each stage of the selection process
    val backAicList = new VectorD(backCols.size)
    val backBicList = new VectorD(backCols.size)

    for k <- 0 until backCols.size do
        val backSubCols = backCols.slice(0, k + 1)                                   // Subset top k+1 features
        val backxSub = x(?, backSubCols)

        val subNN= NeuralNet_3L.rescale (backxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, backQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        backAicList(k) = backQof.col(0)(13)                                                 // Store AIC
        backBicList(k) = backQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val backColsArray = backCols.toArray
    val backColsArrayD = backColsArray.map(_.toDouble)
    val backColsVecD = VectorD(backColsArrayD)                                       // Convert for display
    val backAicBicMatrix = MatrixD(backColsVecD, backAicList, backBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $backAicBicMatrix")

    // Generate and save plots
    if backCols.nonEmpty && backRSq.dim > 0 && backRSq.dim2 > 0 then
        val backRSqPlot = new PlotM(null, backRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Backward Elimination on Insurance Charges", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val backRSqFileName = s"Insurance_P2_Feature_Selection/Scalation_3L_rSq_Backward.png"
        savePlot(backRSqFileName, backRSqPlot, 2.1)
        
        val backAicBicPlot = new PlotM(null, MatrixD(backAicList, backBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Backward Elimination on Insurance Charges", lines = true)
        Thread.sleep(1000)
        val backAicBicFileName = s"Insurance_P2_Feature_Selection/Scalation_3L_AIC_BIC_Backward.png"
        savePlot(backAicBicFileName, backAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Backward Elimination: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val backColsList = backColsArray.toList
    var backColNames = List.fill(backCols.size)("")
    for i <- 0 until backCols.size do
        backColNames = backColNames.updated(i, xFname(backColsList(i)))
    end for

    mod = NeuralNet_3L.rescale (x, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)

    val (stepCols, stepRSq) = mod.stepwiseSelAll()

    // Calculate AIC and BIC for each stage of the selection process
    val stepAicList = new VectorD(stepCols.size)
    val stepBicList = new VectorD(stepCols.size)

    for k <- 0 until stepCols.size do
        val stepSubCols = stepCols.slice(0, k + 1)                                   // Subset top k+1 features
        val stepxSub = x(?, stepSubCols)

        val subNN= NeuralNet_3L.rescale (stepxSub, yy, f = actfHidden, fname = xFname, nz = nz_, f1 = actfOut)
        val (_, stepQof) = subNN.trainNtest()()

        // Close all open windows to free up memory
        for w <- Window.getWindows do
            w.dispose()
        end for
        
        stepAicList(k) = stepQof.col(0)(13)                                                 // Store AIC
        stepBicList(k) = stepQof.col(0)(14)                                                 // Store BIC
    end for

    banner("AIC/BIC History")
    val stepColsArray = stepCols.toArray
    val stepColsArrayD = stepColsArray.map(_.toDouble)
    val stepColsVecD = VectorD(stepColsArrayD)                                       // Convert for display
    val stepAicBicMatrix = MatrixD(stepColsVecD, stepAicList, stepBicList)             // Top row: added variable index
    println(s"AIC/BIC History:      $stepAicBicMatrix")

    // Generate and save plots
    if stepCols.nonEmpty && stepRSq.dim > 0 && stepRSq.dim2 > 0 then
        val stepRSqPlot = new PlotM(null, stepRSq.ᵀ, Regression.metrics, s"R^2 vs n for 3 Layer Neural Network Stepwise Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)                                                         // Pause to allow Java Swing to render
        val stepRSqFileName = s"Insurance_P2_Feature_Selection/Scalation_3L_rSq_Stepwise.png"
        savePlot(stepRSqFileName, stepRSqPlot, 2.1)
        
        val stepAicBicPlot = new PlotM(null, MatrixD(stepAicList, stepBicList), Array("AIC",  "BIC"), s"AIC/BIC vs n for 3 Layer Neural Network Stepwise Selection on Insurance Charges", lines = true)
        Thread.sleep(1000)
        val stepAicBicFileName = s"Insurance_P2_Feature_Selection/Scalation_3L_AIC_BIC_Stepwise.png"
        savePlot(stepAicBicFileName, stepAicBicPlot, 2.1)
    else
        println(s"Skipping plots for 3L NN Forward Selection: No features were selected.")
    end if

    // Close all open windows to free up memory
    for w <- Window.getWindows do
        w.dispose()
    end for

    // Map column indices to feature names
    val stepColsList = stepColsArray.toList
    var stepColNames = List.fill(stepCols.size)("")
    for i <- 0 until stepCols.size do
        stepColNames = stepColNames.updated(i, xFname(stepColsList(i)))
    end for

    println (s"Scalation Forward Selection Order: $forColNames")
    println (s"Scalation Backward Elimination Reversed Order: $backColNames")
    println (s"Scalation Stepwise Selection Order: $stepColNames")

end P2Insurance3LFeatureSelection