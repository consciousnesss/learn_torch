require 'torch';
require 'image';
require 'nn';

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


trainData, testData = prepareData()

-- print out results
local channels = {'y', 'u', 'v' }
for i,channel in ipairs(channels) do
   trainMean = trainData.data[{ {},i }]:mean()
   trainStd = trainData.data[{ {},i }]:std()

   testMean = testData.data[{ {},i }]:mean()
   testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end


first256Samples_y = trainData.data[{ {1,256},1 }]
first256Samples_u = trainData.data[{ {1,256},2 }]
first256Samples_v = trainData.data[{ {1,256},3 }]
image.display(first256Samples_y)
image.display(first256Samples_u)
image.display(first256Samples_v)
