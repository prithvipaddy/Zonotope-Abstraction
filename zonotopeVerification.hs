
import Data.List

type Vector = [Double]
type Layer = ([Vector],Vector, Bool) -- fst = list of list of weights, snd = list of bias, third = apply relu or not
type Zonotope = [[Double]]  -- [[3,1,0], [1,0.5,0.5]] = Zonotope: 3 + 1e1, 1 + 0.5e1 + 0.5e2

getCenter :: Zonotope -> Vector
getCenter = map head

getGenerators :: Zonotope -> [Vector]
getGenerators z = transpose (map tail z)

-- Rebuild a Zonotope from a center and list of generators
buildZonotope :: Vector -> [Vector] -> Zonotope
buildZonotope center generators = zipWith (:) center (transpose generators)

vectorAdd :: Vector -> Vector -> Vector
vectorAdd = zipWith (+)

scalarMult :: Double -> Vector -> Vector
scalarMult s = map (* s)

zonotopeAdd :: Zonotope -> Zonotope -> Zonotope
zonotopeAdd = zipWith (zipWith (+))

-- abstract transformer for relu function
upper :: Zonotope -> [Double]
upper z =
    let
        c = getCenter z
        gs = getGenerators z
    in
        zipWith (+) c (map (sum . map abs) (transpose gs))

lower :: Zonotope -> [Double]
lower z =
    let
        c = getCenter z
        gs = getGenerators z
    in
        zipWith (-) c (map (sum . map abs) (transpose gs))
 -- IMPLEMENT ABSOLUTE VALUE TO FIND CORRECT UPPER
-- MAKE EVERY TRANSFORMER FROM 1D ZONOOTOPE TO 1D 
-- CHECK FOR CASES WHERE THE NUMBER OF GENERATORS DONT MATCH IN 2 ZONOTOPES
-- CHECK FOR CASES WHERE THERE IS LESS DIMENSIONS IN THE INPUT ZONOTOPE THAN THERE ARE INPUT NODES
-- Eg. input zonotope = [[2,4]], first hidden layer = ([[2,0],[1,1]],[1,-1],True). Here, the hidden layer has weights for 2 nodes
-- i.e. in [1,1], it takes 1 x1 + 1 x2, but there is only an x1 which is [2,4].

-- fst = lambda for each dimension of the zonotope, snd = n
findLambdaAndNRelu :: Zonotope -> ([Double],[Double])
findLambdaAndNRelu z =
    let
        u = upper z
        l = lower z
        lambda = zipWith (\ui li -> ui / (ui - li)) u l
        n = zipWith (\ui lambdai -> (ui * (1 - lambdai)) / 2) u lambda
    in
        (lambda,n)

scaleByLambdaRelu :: Zonotope -> Zonotope
scaleByLambdaRelu z =
    let
        lambda = fst (findLambdaAndNRelu z)
        c = getCenter z
        gs = getGenerators z
        scaledC = zipWith (*) lambda c
        scaledGs = map (zipWith (*) lambda) gs
    in
        buildZonotope scaledC scaledGs

-- CONSTRAINT - ZONOTOPE INPUT SHOULD BE 1D
composeLambdaAndNRelu :: Zonotope -> Zonotope
composeLambdaAndNRelu z =
    let
        lambdaScaled = scaleByLambdaRelu z
        lengthOfN = length (head z) + 1
        n = snd (findLambdaAndNRelu z)
    in
        zonotopeAdd (map (++ [0]) lambdaScaled) (nZonotope lengthOfN (head n))

nZonotope :: Int -> Double -> [[Double]]
nZonotope i n | i <= 0 = []
            | i == 1 = [[n]]
            | otherwise = [[n] ++ replicate (i-2) 0 ++ [n]]

applyRelu :: Zonotope -> Zonotope
applyRelu = map (\row -> let singleZonotope = [row]
                         in head (composeLambdaAndNRelu singleZonotope))

-- Apply a linear transformation to a zonotope
linearTransform :: [Vector] -> Vector -> Zonotope -> Zonotope
linearTransform weights bias z =
    let
        c = getCenter z
        gs = getGenerators z
        -- Apply the matrix multiplication to the center and each generator
        transform v = map (sum . zipWith (*) v) weights
        newCenter = vectorAdd (transform c) bias
        newGenerators = map transform gs
    in
        buildZonotope newCenter newGenerators

applyNetwork :: [Layer] -> Zonotope -> Zonotope
applyNetwork layers z = foldl applyLayer z layers

applyLayer :: Zonotope -> Layer -> Zonotope
applyLayer acc (w, b, applyReLU) =
        let transformed = linearTransform w b acc
        in if applyReLU then applyRelu acc else transformed

-- Test data
testZ1 :: Zonotope
testZ1 = [[1, 0.5, 0], [1, 0, 0.5]]

hLayer1 :: Layer
hLayer1 = ([[2,0],[1,1]],[1,-1],False)

reluLayer :: Layer
reluLayer = ([[]],[],True)

outputLayer :: Layer
outputLayer = ([[1,-1]],[0.5],False)

layers :: [Layer]
layers = [hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,hLayer1,reluLayer,outputLayer]

testRelu1 :: Zonotope
testRelu1 = [[2,4]]  -- Zonotope: 2 + 4e1

testRelu2 :: Zonotope
testRelu2 = [[3,1,0], [1,0.5,0.5]]  -- Zonotope: 3 + 1e1, 1 + 0.5e1 + 0.5e2

-- -- type for zonotopes, which consist of a center and a list of generators
-- data Zonotope = Zonotope { center :: Vector, generators :: [Vector] }
--   deriving (Show)

-- vectorAdd :: Vector -> Vector -> Vector
-- vectorAdd = zipWith (+)

-- scalarMult :: Double -> Vector -> Vector
-- scalarMult s = map (* s)

-- zonotopeAdd :: Zonotope -> Zonotope -> Zonotope
-- zonotopeAdd (Zonotope c1 g1) (Zonotope c2 g2) = Zonotope (zipWith (+) c1 c2) (zipWith (zipWith (+)) g1 g2)

-- relu :: Vector -> Vector
-- relu = map (max 0)

-- -- abstract transformer for relu function
-- upper :: Zonotope -> [Double]
-- upper (Zonotope xs xss) = zipWith (+) xs (map sum (transpose xss))

-- lower :: Zonotope -> [Double]
-- lower (Zonotope xs xss) = zipWith (-) xs (map sum (transpose xss))

-- -- fst = lambda for each dimension of the zonotope, snd = n
-- findLambdaAndNRelu :: Zonotope -> ([Double],[Double])
-- findLambdaAndNRelu (Zonotope c gs) =
--     let
--         u = upper (Zonotope c gs)
--         l = lower (Zonotope c gs)
--         lambda = zipWith (\ui li -> ui / (ui - li)) u l
--         n = zipWith (\ui lambdai -> (ui * (1 - lambdai))/2) u lambda
--     in
--         (lambda,n)

-- scaleByLambdaRelu :: [Double] -> Zonotope -> Zonotope
-- scaleByLambdaRelu lambda (Zonotope c gs) =
--     let
--         (scaledC ,scaledGs) = (zipWith (*) lambda c, map (zipWith (*) lambda) gs)
--     in
--         Zonotope scaledC (map (++ [0]) scaledGs)


-- composeLambdaAndNRelu :: Zonotope -> ([Double],[Double]) -> Vector
-- composeLambdaAndNRelu (Zonotope c gs) (lambda,n) =
--     let
--         lambdaScaled = scaleByLambdaRelu lambda (Zonotope c gs)
--         lengthOfN = 1 + length gs
--     in


-- applyRelu :: Zonotope -> Zonotope
-- applyRelu (Zonotope c gs) = Zonotope (relu c) (map relu gs)


-- -- Apply a linear transformation to a zonotope
-- linearTransform :: [Vector] -> Vector -> Zonotope -> Zonotope
-- linearTransform weights bias (Zonotope c gs) =
--     let
--         -- Apply the matrix multiplication to the center and each generator
--         transform v = map (sum . zipWith (*) v) weights
--         newCenter = vectorAdd (transform c) bias
--         newGenerators = map transform gs
--     in
--         Zonotope newCenter newGenerators


-- applyNetwork :: [([[Double]], Vector)] -> Zonotope -> Zonotope
-- applyNetwork layers z = foldl (\acc (w, b) -> applyRelu (linearTransform w b acc)) z layers


-- testZ1 :: Zonotope
-- testZ1 = Zonotope [1,1] [[0.5,0],[0,0.5]]

-- hLayer1 :: Layer
-- hLayer1 = ([[2,0],[1,1]],[1,-1])

-- outputLayer :: Layer
-- outputLayer = ([[1,-1]],[0.5])

-- layers :: [Layer]
-- layers = [hLayer1,outputLayer]

-- testRelu1 :: Zonotope
-- testRelu1 = Zonotope [2] [[4]]
-- testRelu2 :: Zonotope
-- testRelu2 = Zonotope [3, 1] [[1,0.5], [0,0.5]]