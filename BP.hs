module BP (
    outputGradient,
    layerGradient,
    backpropagate,
    updateWeights,
    trainEpoch,
    trainNetwork
) where

import NN (feedForward, reluDerivative, matMul, Matrix, Vector, Layer, Network)
import Data.List (transpose, foldl')

zipWithLengthCheck :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWithLengthCheck f xs ys
  | length xs /= length ys = error $ "zipWithLengthCheck: xs and ys must have same length. xs: " ++ show (length xs) ++ ", ys: " ++ show (length ys)
  | otherwise = zipWith f xs ys

-- | Calculate the gradient of the loss with respect to the output
outputGradient :: (Floating a) => Vector a -> Vector a -> Vector a
outputGradient = zipWithLengthCheck (-)

-- | Calculate the gradient of a layer's weights and biases.
-- | The gradient tells how sensitive the activation/cost function is to each weight and bias.
-- | Output of the activation function is a neuron, i.e., the input to the next layer.
-- | One way to think of it is that the gradient is the slope of the activation/cost function at a given point
-- | it tells us how much the activation/cost function changes when the weight/bias changes
-- | i.e., the relative importance of each weight/bias on the output
layerGradient :: (Floating a, Num a, Ord a) => Vector a -> Vector a -> Vector a -> Layer a
layerGradient inputs delta outputs = (weightGradients, activationGradients)
    where
    activationGradients = zipWithLengthCheck (*) delta (map reluDerivative outputs)
    -- weightGrads[i][j] = inputs[i] * activationGradients[j]
    weightGradients = [[inp * d | d <- activationGradients] | inp <- inputs]

-- | Backpropagate error through the network
backpropagate :: (Floating a, Num a, Ord a) => Vector a -> Vector a -> Network a -> Network a
backpropagate input target network =
  -- 1. Forward pass: Calculate inputs and outputs for each layer
  let allOutputs = scanl (\inp layer -> feedForward inp [layer]) input network
      layerInputs = init allOutputs
      layerOutputs = tail allOutputs

      -- 2. Backward pass (using scanr)
      -- Initial delta for the output layer
      deltaL = outputGradient (last layerOutputs) target

      -- Prepare data for scanr: [(O0, L1), (O1, L2), ..., (O_{N-2}, L_{N-1})]
      -- We need the output of layer 'i' and the weights of layer 'i+1'
      zippedData = zip (init layerOutputs) (tail network)

      -- Step function for scanr: Calculates delta_i using delta_{i+1}
      deltaStep (currentOutput, (nextWeights, _)) nextDelta =
        let weightedDelta = head $ matMul [nextDelta] (transpose nextWeights)
            currentDelta = zipWith (*) weightedDelta (map reluDerivative currentOutput)
        in currentDelta

      -- Compute all deltas [delta_0, ..., delta_{N-2}, deltaL]
      -- scanr processes from right-to-left, accumulating results
      allDeltas = scanr deltaStep deltaL zippedData

      -- 3. Calculate gradients for each layer
  in zipWith3 layerGradient layerInputs allDeltas layerOutputs

-- | Update weights and biases using gradient descent
updateWeights :: (Floating a, Num a, Ord a) => a -> Network a -> Network a -> Network a
updateWeights learningRate network gradients = zipWithLengthCheck updateLayer network gradients
  where
    updateLayer (weights, biases) (weightGrads, biasGrads) =
      let -- weights are a matrix
          newWeights = applyGradientDescent zipWithLengthCheck weights weightGrads
          -- biases are a list  
          newBiases = applyGradientDescent id biases biasGrads
          applyGradientDescent f = zipWithLengthCheck (f $ \p g -> p - learningRate * g)
      in (newWeights, newBiases)

-- | Train the network for one epoch
-- | One epoch is one full pass through the training data
trainEpoch :: (Floating a, Num a, Ord a) => a -> [(Vector a, Vector a)] -> Network a -> Network a
trainEpoch learningRate trainingData network = foldl' step network trainingData
  where
    step currentNetwork (input, target) =
      let gradients = backpropagate input target currentNetwork
      in updateWeights learningRate currentNetwork gradients

-- | Train the network for multiple epochs
trainNetwork :: (Floating a, Num a, Ord a) => Int -> a -> [(Vector a, Vector a)] -> Network a -> Network a
trainNetwork epochs learningRate trainingData initialNetwork =
  foldl' (\network _ -> trainEpoch learningRate trainingData network) initialNetwork [1..epochs] 