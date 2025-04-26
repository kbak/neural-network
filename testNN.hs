import NN (feedForward, generateRandomMatrix, createNetwork, mse)

main :: IO ()
main = do
  -- Example network with input layer (2), two hidden layers (10,8) and output layer (4)
  -- The numbers indicate the number of neurons in each layer
  let networkShape = [2,10,8,4]
  network <- createNetwork networkShape

  -- Generate random input matching input layer size
  input <- head <$> generateRandomMatrix 1 (head networkShape) :: IO [Double]
  putStrLn $ "Input  : " ++ show input

  -- Feed forward through the network
  let output = feedForward input network

  -- Print results
  putStrLn $ "Output : " ++ show output