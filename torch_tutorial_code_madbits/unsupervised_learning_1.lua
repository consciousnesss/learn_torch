require 'torch';
require 'nn';
require 'unsup';  -- run 'luarocks install unsup'
require 'xlua';
require 'image';
require 'optim';


function getData(datafile, inputsize)
    -- this function apparently selects a random patches from dataset images.
    -- It returns patches that have enough variations in them.
    local std = 0.2
    local data = torch.load(datafile, 'ascii')
    local dataset = {}

    local nsamples = data:size(1)
    local nrows = data:size(2)
    local ncols = data:size(3)

    function dataset:size()
        return nsamples
    end
    function dataset:selectPatchWithEnoughStd(nrow, ncol)
        while true do
            -- image index
            local i = math.ceil(torch.uniform(1e-12, nsamples))
            local im = data:select(i, 1)
            local ri = math.ceil(torch.uniform(1e-12, nrows-nrow))
            local ci = math.ceil(torch.uniform(1e-12, ncols-ncol))
            local patch = im:narrow(1, ri, nrow)
            patch = patch:narrow(2, ci, ncol)
            local patchstd = patch:std()
            if patchstd > std then
                return patch, i, im
            end
        end
    end

    local dsample = torch.Tensor(inputsize*inputsize)

    function dataset:conv()
        dsample = torch.Tensor(1, inputsize, inputsize)
    end

    setmetatable(datset, {__index = function(self, index)
        local sample, i, im = self:selectPatchWithEnoughStd(inputsize, inputsize, 0.2)
        dsample:copy(sample)
        return {dsample, dsample, im}
    end})

    return dataset
end



function createLinearAutoencoder(imageSize, outputSize)
    local inputSize = imageSize*imageSize

    local encoder = nn.Sequential()
    encoder:add(nn.Linear(inputSize, outputSize))
    encoder:add(nn.Tanh())

    local decoder = nn.Sequential()
    decode:add(nn.Linear(outputSize, inputSize))

    -- weights sharing
    decoder:get(1).weight = encoder:get(1).weight:t()
    decoder:get(1).gradWeight = encoder:get(1).gradWeight:t()

    return unsup.AutoEncoder(encoder, decoder, 1)
end


function createConvAutoencoder(imageSize, filtersIn, filtersOut, kernelSize)
    local kw, kh = kernelSize, kernelSize
    local iw, ih = imageSize, imageSize

    -- connection table:
    local conntable = nn.tables.full(filtersIn, filtersOut)
    local decodertable = conntable:clone()
    decodertable[{ {},1 }] = conntable[{ {},2 }]
    decodertable[{ {},2 }] = conntable[{ {},1 }]

    local outputFeatures = conntable[{ {}, 2}]:max()

    local encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolutionMap(conntable, kw, kh, 1, 1))
    encoder:add(nn.Tanh())
    encoder:add(nn.Diag(outputFeatures))

    local decoder = nn.Sequential()
    -- it is not clear to me if the intent to get deconvolution here..
    decoder:add(nn.SpatialFullConvolutionMap(decodertable, kw, kh, 1, 1))

    return unsup.AutoEncoder(encoder, decoder, 1)
end


function createPredictiveSparseDecompositionAutoencoder(imageSize, outputSize)
    local inputSize = imageSize*imageSize

    local encoder = nn.Sequential()
    encoder:add(nn.Linear(inputSize,outputSize))
    encoder:add(nn.Tanh())
    encoder:add(nn.Diag(outputSize)) -- layer of multiplicative gains. Not sure why.

    -- decoder is L1 solution, relies on FISTA to find the optimal sparse code. FISTA is available in the optim package.
    -- more here: http://code.madbits.com/wiki/doku.php?id=tutorial_unsupervised_1_models
    local decoder = unsup.LinearFistaL1(inputSize, outputSize, 1)

    -- PSD autoencoder
    return unsup.PSD(encoder, decoder, 1)
end


function trainModuleSGD(module, dataset, inputsize, outputsize, modelname)
    local x,dl_dx = module:getParameters()
    local maxiter = 2
    local batchsize = 1
    local statinterval = 2
    local eta = 2e-3
    local etadecay = 1e-5
    local momentum = 0

    local err = 0
    local iter = 1

    for t = 1,maxiter,batchsize do
        xlua.progress(iter*batchsize, statinterval)

        local inputs = {}
        local targets = {}
         -- create batch
        for i=t, t+batchsize do
            local sample = dataset[i]
            local input = sample[1]:clone()
            local target = sample[2]:clone()
            table.insert(inputs, input)
            table.insert(targets, target)
        end


        local feval = function ()
            local f = 0
            dl_dx:zero()

            for i=1,#inputs do
                -- compute output of the module costs
                f = f + module:updateOutput(inputs[i], targets[i])

                -- compute model input gradients
                module:updateGradInput(inputs[i], targets[i])
                -- compute model gradients to the parameters (will be accumulated in dl_dx)
                module:accGradParameters(inputs[i], targets[i])
            end

            dl_dx:div(#inputs)
            f = f/#inputs

            return f, dl_dx
        end


        sgdconf = sgdconf or {
            learningRate = eta,
            learningRateDecay = etadecay,
            learningRates = etas,
            momentum = momentum}
        _,fs = optim.sgd(feval, x, sgdconf)
        err = err + fs[1]*batchsize -- so that err is indep of batch size

        -- normalize
        if modelname:find('psd') then
          module:normalize()
        end

        -- compute statistics / report error
        if iter*batchsize >= statinterval then
            -- report
            print('==> iteration = ' .. t .. ', average loss = ' .. err/statinterval)

            -- get weights
            eweight = module.encoder.modules[1].weight
            if module.decoder.D then
                dweight = module.decoder.D.weight
            else
                dweight = module.decoder.modules[1].weight
            end

            -- reshape weights if linear matrix is used
            if modelname:find('linear') then
                dweight = dweight:transpose(1,2):unfold(2,inputsize,inputsize)
                eweight = eweight:unfold(2,inputsize,inputsize)
            end

            -- render filters
            dd = image.toDisplayTensor{
                input=dweight,
                padding=2,
                nrow=math.floor(math.sqrt(outputsize)),
                symmetric=true}
            de = image.toDisplayTensor{
                input=eweight,
                padding=2,
                nrow=math.floor(math.sqrt(outputsize)),
                symmetric=true}

            -- live display
            print('Decoder filters')
            image.display(dd)
            print('Encoder filters')
            image.display(de)

            -- reset counters
            err = 0; iter = 0
        end
    end
end


dataurl = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/tr-berkeley-N5K-M56x56-lcn.ascii'
filename = paths.basename(dataurl)
if not paths.filep(filename) then
   os.execute('wget ' .. dataurl)
end

inputsize = 25
dataset = getData(filename, inputsize)

