require 'nn';
require 'image';


function trainCifar10()
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip -n cifar10torchsmall.zip')
    local trainset = torch.load('cifar10-train.t7')
    local testset = torch.load('cifar10-test.t7')
    local classes = {'airplane', 'automobile', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }

    setmetatable(trainset,
        {__index = function(t, i)
                        return {t.data[i], t.label[i]}
                    end }
    );
    function trainset:size()
        return self.data:size(1)
    end

    trainset.data = trainset.data:double()

    print(trainset:size(1))


--    image.display(trainset[33][1])

    local mean = {}
    local std = {}

    for i=1,3 do
        mean[i] = trainset.data[{{}, {i}, {}, {}}]:mean()
        print('Channel ' .. i .. ', mean: ' .. mean[i])
        trainset.data[{{}, {i}, {}, {}}]:add(-mean[i])

        std[i] = trainset.data[{{}, {i}, {}, {}}]:std()
        print('Channel ' .. i .. ', stdv: ' .. std[i])
        trainset.data[{{}, {i}, {}, {}}]:div(std[i])
    end


    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
    net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
    net:add(nn.SpatialConvolution(6, 16, 5, 5))
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    -- Again, it seems that activation functions are missing
    net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
    net:add(nn.Linear(120, 84))
    net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
    net:add(nn.LogSoftMax())

    local criterion = nn.ClassNLLCriterion()

    local trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 5
    trainer:train(trainset)

    testset.data = testset.data:double()
    for i=1,3 do
        testset.data[{{}, {i}, {}, {}}]:add(-mean[i])
        testset.data[{{}, {i}, {}, {}}]:div(std[i])
    end

    local predicted = net:forward(testset.data[100])
    predicted = predicted:exp()
    for i=1,predicted:size(1) do
        print(classes[i], predicted[i])
    end

    local class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    for i=1,testset.data:size(1) do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            class_performance[groundtruth] = class_performance[groundtruth] + 1
        end
    end

    for i=1,#classes do
        print(classes[i], 100*class_performance[i]/1000 .. ' %')
    end

    print('now train on GPU..')
    net = net:cuda()
    criterion = criterion:cuda()
    trainset.data = trainset.data:cuda()
    testset.data = testset.data:cuda()
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = 0.001
    trainer.maxIteration = 5
    trainer:train(trainset)

    local class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    for i=1,testset.data:size(1) do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            class_performance[groundtruth] = class_performance[groundtruth] + 1
        end
    end

    for i=1,#classes do
        print(classes[i], 100*class_performance[i]/1000 .. ' %')
    end
end


trainCifar10()
