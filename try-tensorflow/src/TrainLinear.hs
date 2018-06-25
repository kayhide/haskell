module TrainLinear (run) where

import           ClassyPrelude

import           System.Random          (randoms, newStdGen)
import qualified TensorFlow.Core        as TF
import qualified TensorFlow.GenOps.Core as TF
import qualified TensorFlow.Minimize    as TF
import qualified TensorFlow.Ops         as TF hiding (initializedVariable)
import qualified TensorFlow.Variable    as TF

run :: IO ()
run = do
  gen <- newStdGen

  -- Generate data where `y = x*3 + 8`.
  let xs = take 100 $ randoms gen
  let ys = (+ 8) . (* 3) <$> xs

  -- Fit linear regression model.
  (w, b) <- fit xs ys
  putStrLn "y = w * x + b"
  putStrLn $ "w: " <> tshow w
  putStrLn $ "b: " <> tshow b

fit :: [Float] -> [Float] -> IO (Float, Float)
fit xs ys = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TF.vector xs
        y = TF.vector ys
    -- Create scalar variables for slope and intercept.
    w <- TF.initializedVariable 0
    b <- TF.initializedVariable 0
    -- Define the loss function.
    let yHat = (x `TF.mul` TF.readValue w) `TF.add` TF.readValue b
        loss = TF.square (yHat `TF.sub` y)
    -- Optimize with gradient descent.
    trainStep <- TF.minimizeWith (TF.gradientDescent 0.001) loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readValue w, TF.readValue b)
    return (w', b')
