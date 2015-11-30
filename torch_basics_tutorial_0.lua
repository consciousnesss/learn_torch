require 'nn';
require 'image';

local basicsTutorialTest = {}
local tester = torch.Tester()
tester.countasserts=0


function torch.Tester:assertVectorsEq(vec1, vec2)
    if #vec1 ~= #vec2 then
        self:asserteq(#vec1, #vec2, 'Vectors have different length')
        return
    end

    for i = 1,#vec1 do
        if vec1[i] ~= vec2[i] then
            self:asserteq(vec1[i], vec2[i], 'Elements are not equal')
            return
        end
    end
end


function basicsTutorialTest.multiply()
    local a = torch.rand(5, 3)
    local b = torch.rand(3, 4)
    local c = a*b

    tester:asserteq(c:nDimension(), 2)
    tester:assertVectorsEq(torch.LongStorage({5, 4}), c:size())
end


function basicsTutorialTest.buildLenet()
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(1, 6, 5, 5))
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    net:add(nn.SpatialConvolution(6, 16, 5, 5))
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    net:add(nn.View(16*5*5))
    -- it seems to me that the following layers supposed to have activation functions..
    net:add(nn.Linear(16*5*5, 120))
    net:add(nn.Linear(120, 84))
    net:add(nn.Linear(84, 10))
    net:add(nn.LogSoftMax())
    tester:asserteq(net:size(), 9)
end


tester:add(basicsTutorialTest)
tester:run()
if tester.errors[1] then os.exit(1) end
