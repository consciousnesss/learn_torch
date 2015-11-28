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
        data = loadedData.X:transpose(3, 4),
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
    for i,channel in ipairs(channels) do
        local mean = trainData.data[{ {}, i, {}, {} }]:mean()
        local std = trainData.data[{ {}, i, {}, {} }]:std()
        trainData.data[{ {}, i, {}, {} }]:add(-mean)
        trainData.data[{ {}, i, {}, {} }]:div(std)

        testData.data[{ {}, i, {}, {} }]:add(-mean)
        testData.data[{ {}, i, {}, {} }]:div(std)
    end
end



trainFile, testFile = downloadHousenumbersData()
local trainSize = 10000
local testSize = 2000
trainData = loadData(trainFile, trainSize)
testData = loadData(testFile, testSize)

normalizeData(trainData, testData)
