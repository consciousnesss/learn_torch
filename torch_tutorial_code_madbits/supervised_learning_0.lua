require 'torch';
require 'image';
require 'nn';
require 'optim';
require 'xlua';

function downloadHousenumbersData()
    local url = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/housenumbers/'
    local trainFile = 'train_32x32.t7'
    local testFile = 'test_32x32.t7'
    local extraFile = 'extra_32x32.t7'

    if not paths.filep(trainFile) then
        os.execute('wget ' .. url .. trainFile)
    end

    if not paths.filep(testFile) then
        os.execute('wget ' .. url .. testFile)
    end

    return trainFile, testFile
end


function loadData(filename, size)
    local loadedData = torch.load(filename, 'ascii')
    local dataDesc = {
        data = loadedData.X:transpose(3, 4)[{{1, size}, {}, {}, {}}],
        labels = loadedData.y[1],
        size = function() return size end
    }
    dataDesc.data = dataDesc.data:float()
    for i = 1,size do
        dataDesc.data[i] = image.rgb2yuv(dataDesc.data[i])
    end

    return dataDesc
end


function normalizeData(trainData, testData)
    local channels = {'y', 'u', 'v' }
    for i in ipairs(channels) do
        local mean = trainData.data[{ {}, i, {}, {} }]:mean()
        local std = trainData.data[{ {}, i, {}, {} }]:std()
        trainData.data[{ {}, i, {}, {} }]:add(-mean)
        trainData.data[{ {}, i, {}, {} }]:div(std)

        testData.data[{ {}, i, {}, {} }]:add(-mean)
        testData.data[{ {}, i, {}, {} }]:div(std)
    end

    -- local normalization
    local neirborhood = image.gaussian1D(13)
    local normalization = nn.SpatialContrastiveNormalization(1, neirborhood, 1):float()

    -- normalize only Y
    for i = 1,trainData:size() do
      trainData.data[{ i,{1},{},{} }] = normalization:forward(trainData.data[{ i,{1},{},{} }])
    end
    for i = 1,testData:size() do
      testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
    end
end



function prepareData()
    local preparedTrainFile = 'prepared_train_32x32.t7'
    local preparedTestFile = 'prepared_test_32x32.t7'
    if paths.filep(preparedTrainFile) and paths.filep(preparedTestFile) then
        print('Loading prepared dataset..')
        local trainData = torch.load(preparedTrainFile)
        local testData = torch.load(preparedTestFile)
        return trainData, testData
    else
        local trainFile, testFile = downloadHousenumbersData()
        local trainSize = 10000
        local testSize = 2000
        local trainData = loadData(trainFile, trainSize)
        local testData = loadData(testFile, testSize)

        normalizeData(trainData, testData)

        torch.save(preparedTrainFile, trainData)
        torch.save(preparedTestFile, testData)
        return trainData, testData
    end
end


function createLinear(nInputs, nOutputs)
    local model = nn.Sequential()
    model:add(nn.Reshape(nInputs))
    model:add(nn.Linear(nInputs, nOutputs))
    return model
end


function createMLP(nInputs, nOutputs)
    local nHidden = nInputs/2
    local model = nn.Sequential()
    model:add(nn.Reshape(nInputs))
    model:add(nn.Linear(nInputs, nHidden))
    model:add(nn.Tanh())
    model:add(nn.Linear(nHidden, nOutputs))
end


function createConvNet(nOutputs)
    local nstates = {64, 64, 128}
    local filterSize = 5
    local poolSize = 2
    local nFeatures = 3
    local normkernel = image.gaussian1D(7)

    local model = nn.Sequential()

    -- stage 1
    model:add(nn.SpatialConvolutionMM(nFeatures, nstates[1], filterSize, filterSize))
    model:add(nn.Tanh())
    model:add(nn.SpatialLPPooling(nstates[1], 2, poolSize, poolSize, poolSize, poolSize))
    model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

    -- stage 2
    model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filterSize, filterSize))
    model:add(nn.Tanh())
    model:add(nn.SpatialLPPooling(nstates[2], 2, poolSize, poolSize, poolSize, poolSize))
    model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

    -- stage 3
    model:add(nn.Reshape(nstates[2]*filterSize*filterSize))
    model:add(nn.Linear(nstates[2]*filterSize*filterSize, nstates[3]))
    model:add(nn.Tanh())
    model:add(nn.Linear(nstates[3], nOutputs))
    return model
end

function train(trainData, batchSize)
    epoch = epoch or 1
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    shuffle = torch.randperm(trainData:size())

    for t = 1,trainData:size(),batchSize do
        xlua.progress(t, trainData:size())

        local inputs = {}
        local targets = {}

        -- create random batch
        for i = t,math.min(t+batchSize-1, trainData:size()) do
            local input = trainData.data[shuffle[i]]:double()
            local target = trainData.labels[shuffle[i]]

            table.insert(inputs, input)
            table.insert(targets, target)
        end

        local computeGradientOnBatch = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            gradParameters:zero()

            local f = 0

            for i =1,#inputs do
                local output = model:forward(inputs[i])
                local err = criterion:forward(output, targets[i])
                f = f + err

                -- compute gradient but do not change parameters
                local df_do = criterion:backward(output, targets[i])
                model:backward(inputs[i], df_do)

                confusion:add(output, targets[i])
            end

            -- normalize gradients
            gradParameters:div(#inputs)
            f = f/#inputs

            return f, gradParameters
        end


        config = config or {
          learningRate = learningRate,
          weightDecay = weightDecay,
          momentum = momentum,
          learningRateDecay = 1e-7
        }

        optim.sgd(computeGradientOnBatch, parameters, optimState)

    end

    time = sys.clock() - time
    time = time/trainData:size()

    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    print(confusion)

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end


-- test function
function test(testData)
   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   for t = 1,testData:size() do
      -- disp progress
      xlua.progress(t, testData:size())

      local input = testData.data[t]:double()
      local target = testData.labels[t]

      local pred = model:forward(input)
      confusion:add(pred, target)
   end

   -- timing
   time = sys.clock() - time
   time = time / testData:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- next iteration:
   confusion:zero()
end


trainData, testData = prepareData()

nFeatures = 3
width = 32
heigth = 32
nInputs = nFeatures*width*heigth
nOutputs = 10

model = createConvNet(nOutputs)

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()


classes = {'1','2','3','4','5','6','7','8','9','0' }
confusion = optim.ConfusionMatrix(classes)

parameters, gradParameters = model:getParameters()

local learningRate = 1e-3
local weightDecay = 0
local momentum = 0
local batchSize = 1

for epochn=1,2 do
    train(trainData, batchSize)
    test(testData)
end
