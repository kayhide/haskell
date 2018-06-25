module Main where

import           ClassyPrelude

import qualified TrainLinear
import qualified TrainMnist


main :: IO ()
main = do
  putStrLn "Start TrainLinear"
  TrainLinear.run
  putStrLn "Start TrainMnist"
  TrainMnist.run
