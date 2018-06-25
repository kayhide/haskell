module TrainMnist (run) where

import           ClassyPrelude

import           Data.List                           (cycle, genericLength,
                                                      (!!))
import           Data.List.Split                     (chunksOf)
import qualified Data.Vector                         as V

import qualified TensorFlow.Core                     as TF
import qualified TensorFlow.Minimize                 as TF
import qualified TensorFlow.Ops                      as TF hiding
                                                            (initializedVariable,
                                                            zeroInitializedVariable)
import qualified TensorFlow.Variable                 as TF

import           TensorFlow.Examples.MNIST.InputData (testImageData,
                                                      testLabelData,
                                                      trainingImageData,
                                                      trainingLabelData)
import           TensorFlow.Examples.MNIST.Parse     (MNIST, drawMNIST,
                                                      readMNISTLabels,
                                                      readMNISTSamples)


numPixels :: Int64
numPixels = 28*28

numLabels :: Int64
numLabels = 10

-- | Create tensor with random values where the stddev depends on the width.
randomParam :: Int64 -> TF.Shape -> TF.Build (TF.Tensor TF.Build Float)
randomParam width (TF.Shape shape) =
    (`TF.mul` stddev) <$> TF.truncatedNormal (TF.vector shape)
  where
    stddev = TF.scalar (1 / sqrt (fromIntegral width))

-- Types must match due to model structure.
type LabelType = Int32

data Model = Model {
      train :: TF.TensorData Float  -- ^ images
            -> TF.TensorData LabelType
            -> TF.Session ()
    , infer :: TF.TensorData Float  -- ^ images
            -> TF.Session (Vector LabelType)  -- ^ predictions
    , errorRate :: TF.TensorData Float  -- ^ images
                -> TF.TensorData LabelType
                -> TF.Session Float
    }

createModel :: TF.Build Model
createModel = do
    -- Use -1 batch size to support variable sized batches.
    let batchSize = -1
    -- Inputs.
    images <- TF.placeholder [batchSize, numPixels]
    -- Hidden layer.
    let numUnits = 500
    hiddenWeights <-
        TF.initializedVariable =<< randomParam numPixels [numPixels, numUnits]
    hiddenBiases <- TF.zeroInitializedVariable [numUnits]
    let hiddenZ = (images `TF.matMul` TF.readValue hiddenWeights)
                  `TF.add` TF.readValue hiddenBiases
    let hidden = TF.relu hiddenZ
    -- Logits.
    logitWeights <-
        TF.initializedVariable =<< randomParam numUnits [numUnits, numLabels]
    logitBiases <- TF.zeroInitializedVariable [numLabels]
    let logits = (hidden `TF.matMul` TF.readValue logitWeights)
                 `TF.add` TF.readValue logitBiases
    predict <- TF.render @TF.Build @LabelType $
               TF.argMax (TF.softmax logits) (TF.scalar (1 :: LabelType))

    -- Create training action.
    labels <- TF.placeholder [batchSize]
    let labelVecs = TF.oneHot labels (fromIntegral numLabels) 1 0
        loss =
            TF.reduceMean $ fst $ TF.softmaxCrossEntropyWithLogits logits labelVecs
        params = [hiddenWeights, hiddenBiases, logitWeights, logitBiases]
    trainStep <- TF.minimizeWith TF.adam loss params

    let correctPredictions = TF.equal predict labels
    errorRateTensor <- TF.render $ 1 - TF.reduceMean (TF.cast correctPredictions)

    return Model {
          train = \imFeed lFeed -> TF.runWithFeeds_ [
                TF.feed images imFeed
              , TF.feed labels lFeed
              ] trainStep
        , infer = \imFeed -> TF.runWithFeeds [TF.feed images imFeed] predict
        , errorRate = \imFeed lFeed -> TF.unScalar <$> TF.runWithFeeds [
                TF.feed images imFeed
              , TF.feed labels lFeed
              ] errorRateTensor
        }

encodeImageBatch :: [MNIST] -> TF.TensorData Float
encodeImageBatch xs =
  TF.encodeTensorData [genericLength xs, numPixels]
  (fromIntegral <$> mconcat xs)

encodeLabelBatch :: [Word8] -> TF.TensorData LabelType
encodeLabelBatch xs =
  TF.encodeTensorData [genericLength xs]
  (fromIntegral <$> V.fromList xs)

run :: IO ()
run = TF.runSession $ do
    -- Read training and test data.
    trainingImages :: [MNIST] <- liftIO (readMNISTSamples =<< trainingImageData)
    trainingLabels <- liftIO (readMNISTLabels =<< trainingLabelData)
    testImages :: [MNIST] <- liftIO (readMNISTSamples =<< testImageData)
    testLabels <- liftIO (readMNISTLabels =<< testLabelData)

    -- Create the model.
    model <- TF.build createModel

    -- Train.
    let train' (i :: Int, (images, labels)) = do
          train model images labels
          when (i `mod` 100 == 0) $ do
            err <- errorRate model images labels
            liftIO $ putStrLn $ "training error " <> tshow (err * 100)
    let imageBatches = encodeImageBatch <$> chunksOf 100 (cycle trainingImages)
        labelBatches = encodeLabelBatch <$> chunksOf 100 (cycle trainingLabels)
        batches = zip imageBatches labelBatches
    traverse_ train' $ zip [0..1000] batches
    liftIO $ putStrLn ""

    -- Test.
    testErr <- errorRate model (encodeImageBatch testImages)
                               (encodeLabelBatch testLabels)
    liftIO $ putStrLn $ "test error " <> tshow (testErr * 100)

    -- Show some predictions.
    testPreds <- infer model (encodeImageBatch testImages)
    liftIO $ forM_ ([0..3] :: [Int]) $ \i -> do
        putStrLn ""
        putStrLn $ drawMNIST $ testImages !! i
        putStrLn $ "expected " <> tshow (testLabels !! i)
        putStrLn $ "     got " <> tshow (testPreds V.! i)
