import NN (feedForward, createNetwork, mse, printNetwork)
import BP (trainNetwork)
import System.Random (newStdGen)

main :: IO ()
main = do
  -- Define network parameters
  let learningRate = 0.1
      epochs = 1000

  -- Create training data (2-bit binary to one-hot encoding)
  let trainingData@((tdInput, tdTarget):_) = [
                     ([0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
                     ([0.0, 1.0], [0.0, 1.0, 0.0, 0.0]),
                     ([1.0, 0.0], [0.0, 0.0, 1.0, 0.0]),
                     ([1.0, 1.0], [0.0, 0.0, 0.0, 1.0])]

  let nInputs = length tdInput
  let nOutputs = length tdTarget

  -- Create initial layer configurations
  putStrLn "Initial network"
  putStrLn "==============="
  initialNetwork <- createNetwork [nInputs, 10, 8, nOutputs]

  printNetwork initialNetwork

  -- Train the network
  let trainedNetwork = trainNetwork epochs learningRate trainingData initialNetwork

  putStrLn "Trained network"
  putStrLn "==============="
  printNetwork trainedNetwork

  -- Test the trained network
  putStrLn "Testing trained network:"
  mapM_ (\(input, target) -> do
    let output = feedForward input trainedNetwork
    putStrLn $ "Input  : " ++ show input
    putStrLn $ "Target : " ++ show target
    putStrLn $ "Output : " ++ show output
    putStrLn $ "Error  : " ++ show (mse output target)
    putStrLn "") trainingData
