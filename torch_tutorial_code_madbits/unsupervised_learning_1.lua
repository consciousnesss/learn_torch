require 'torch';
require 'nn';
require 'unsup';


function getData(datafile, inputsize, std)
    -- this function apparently selects a random patches from dataset images.
    -- It returns patches that have enough variations in them.
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
    local decoder = unsup.LinearFistaL1(inputSize, outputSize, params.lambda)

    -- PSD autoencoder
    return unsup.PSD(encoder, decoder, params.beta)
end
