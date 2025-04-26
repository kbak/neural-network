module NN (
    reluDerivative,
    matMul,
    generateRandomMatrix,
    createNetwork,
    printNetwork,
    feedForward,
    mse,
    Matrix,
    Vector,
    Layer,
    Network
) where

import Data.List (transpose)
import System.Random (Random, randomRs, newStdGen)
import Text.Printf (printf, PrintfArg)

-- Type Aliases
type Matrix a = [[a]]
type Vector a = [a]
type Layer a = (Matrix a, Vector a) -- (Weights, Biases)
type Network a = [Layer a]

-- | Rectified Linear Unit (ReLU) activation function
-- | returns 0 if the input is negative, otherwise returns the input
relu :: (Ord a, Num a) => a -> a
relu x = max 0 x

-- | Derivative of the ReLU function
reluDerivative :: (Ord a, Num a) => a -> a
reluDerivative x = if x > 0 then 1 else 0

-- | Matrix multiplication (2D lists): a x b
-- | where a is a matrix of size m x n (rows x columns)
-- | and b is a matrix of size n x p (rows x columns).
-- | To match 
matMul :: Num a => Matrix a -> Matrix a -> Matrix a
matMul a b
  | null a || null b = error "matMul: empty matrix"
  | colsA /= rowsB = error $ "matMul: dimension mismatch - a has " ++ show colsA ++ " columns but b has " ++ show rowsB ++ " rows"
  | otherwise = [[ sum $ zipWith (*) ar bc | bc <- transpose b ] | ar <- a ]
  where
    colsA = length (head a)
    rowsB = length b

-- | Calculate the output of a single layer
-- | Each neuron is a function
calculateLayer :: (Floating a, Num a, Ord a) => Vector a -> Layer a -> Vector a
calculateLayer inputs (weights, biases) = layerOutputs
  where
    -- compute weighted sum of inputs. Result is a vector (hence head)
    layerInputs = head $ matMul [inputs] weights
    -- add biases which signify inactivation of the neuron (some threshold)
    -- only when the weighted sum of the inputs is greater than the bias, the neuron is activated
    layerInputsWithBiases = zipWith (+) layerInputs biases
    -- apply activation function to determine how much the neuron is activated
    layerOutputs = map relu layerInputsWithBiases

-- | A feedforward neural network that can handle any number of layers
feedForward :: (Floating a, Num a, Ord a) => Vector a -> Network a -> Vector a
feedForward = foldl calculateLayer

-- | Random weights and biases generator
-- | Uses wider range for ReLU activation
generateRandomMatrix :: (Random a, Num a) => Int -> Int -> IO (Matrix a)
generateRandomMatrix rows cols = do
  g <- newStdGen
  let values = randomRs (-1, 1) g  -- Wider range suitable for ReLU
  return $ take rows (map (take cols) (iterate (drop cols) values))

-- Creates a neural network with specified layer sizes
createNetwork :: [Int] -> IO (Network Double)
createNetwork layerSizes
  | length layerSizes < 2 = error "Network must have at least input and output layers"
  | otherwise = do
    -- Generate weights and biases for each layer pair
    layerPairs <- sequence
      [ do weights <- generateRandomMatrix rows cols :: IO [[Double]]
           biases <- generateRandomMatrix 1 cols :: IO [[Double]]
           return (weights, head biases)
      | (rows, cols) <- zip layerSizes (tail layerSizes)
      ]
    return layerPairs

-- | Loss/cost function (Mean Squared Error)
-- | Measures the average squared difference between the predicted output and the target output
mse :: (Floating a, Num a) => Vector a -> Vector a -> a
mse output target
  | length output /= length target = error "MSE: output and target must have the same length"
  | otherwise = sum $ map (\(o, t) -> (o - t) ** 2) (zip output target)

-- | Pretty print a matrix (list of lists)
-- | Each row is printed on a new line, elements space-separated
-- | Takes a label for the header row.
prettyPrintMatrix :: (PrintfArg a, Show a) => String -> Matrix a -> String
prettyPrintMatrix _ [] = "" -- Handle empty matrix
prettyPrintMatrix headerLabel matrix@(firstRow:_) =
  let numCols = length firstRow
      -- Create header with column indices
      colIndices = [0..numCols-1]
      formattedIndices = map (printf "%10d") colIndices
      -- Format the header label (left-aligned, 7 chars wide)
      formattedHeaderLabel = printf "%-7s" headerLabel 
      header = formattedHeaderLabel ++ unwords formattedIndices
      -- Create separator line based on header width
      lineWidth = length header
      separator = replicate lineWidth '-'
      -- Format data rows with row indices
      formattedRows = zipWith formatRow [0..] matrix
  in unlines (header : separator : formattedRows)
  where
    formatRow :: (PrintfArg a, Show a) => Int -> [a] -> String
    formatRow index row = printf "  %2d | %s" index formattedRowContent
      where
        formattedRowContent = unwords $ map (printf "%10.4f") row

-- | Pretty print a single layer (weights and biases)
prettyPrintLayer :: (PrintfArg a, Show a) => Layer a -> String
prettyPrintLayer (weights, biases) =
    prettyPrintMatrix "\nWeights" weights ++ "\n" ++ prettyPrintMatrix "Biases" [biases]

-- | Pretty print the entire network layer by layer
prettyPrintNetwork :: (PrintfArg a, Show a) => Network a -> String
prettyPrintNetwork network =
    unlines $ zipWith (\i l -> "Layer " ++ show i ++ "\n" ++ prettyPrintLayer l)
    [1..] network
    
-- | Print the network to the console with nice formatting
printNetwork :: (PrintfArg a, Show a) => Network a -> IO ()
printNetwork = putStrLn . prettyPrintNetwork
