--[[
This example shows how to transform an
uploaded image into a torch Tensor.
]]

require 'image'
require 'nn'
require 'nngraph'
util = paths.dofile('util/util.lua')

-- name=onepiece input_nc=1 output_nc=2 preprocess=colorization which_direction=AtoB th web.lua
opt = {
    batchSize = 1,            -- # images in batch
    loadSize = 256,           -- scale images to this size
    fineSize = 256,           --  then crop to this size
    flip=0,                   -- horizontal mirroring data augmentation
    gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
    which_direction = 'BtoA', -- AtoB or BtoA
    phase = 'val',            -- train, val, test ,etc
    preprocess = 'regular',   -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
    aspect_ratio = 1.0,       -- aspect ratio of result images
    name = '',                -- name of experiment, selects which model to run, should generally should be passed on command line
    input_nc = 3,             -- #  of input image channels
    output_nc = 3,            -- #  of output image channels
    serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
    serial_batch_iter = 1,    -- iter into serial image list
    cudnn = 1,                -- set to 0 to not use cudnn (untested)
    checkpoints_dir = './checkpoints', -- loads models from here
    results_dir='./results/',          -- saves results here
    which_epoch = 'latest',            -- which epoch to test? set to 'latest' to use latest cached model
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
opt.nThreads = 1 -- test only works with 1 thread...
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- set seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

opt.netG_name = opt.name .. '/' .. opt.which_epoch .. '_net_G'

-- translation direction
local idx_A = nil
local idx_B = nil
local input_nc = opt.input_nc
local output_nc = opt.output_nc
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local function loadImageChannel(path)
  -- print("load image: ", path)
  local input = image.load(path, 3, 'float')
  input = image.scale(input, loadSize[2], loadSize[2])

  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = input:size(2)
  local iW = input:size(3)
  
  if iH~=oH then     
    h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  end
  
  if iW~=oW then
    w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  end
  if iH ~= oH or iW ~= oW then 
    input = image.crop(input, w1, h1, w1 + oW, h1 + oH)
  end
  
  
  if opt.flip == 1 and torch.uniform() > 0.5 then 
    input = image.hflip(input)
  end
  
--    print(input:mean(), input:min(), input:max())
  local input_lab = image.rgb2lab(input)
--    print(input_lab:size())
--    os.exit()
  local imA = input_lab[{{1}, {}, {} }]:div(50.0) - 1.0
  local imB = input_lab[{{2,3},{},{}}]:div(110.0)
  local imAB = torch.cat(imA, imB, 1)
  assert(imAB:max()<=1,"A: badly scaled inputs")
  assert(imAB:min()>=-1,"A: badly scaled inputs")
  
  return imAB
end

if opt.which_direction=='AtoB' then
  idx_A = {1, input_nc}
  idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
  idx_A = {input_nc+1, input_nc+output_nc}
  idx_B = {1, input_nc}
else
  error(string.format('bad direction %s',opt.which_direction))
end
----------------------------------------------------------------------------

print('checkpoints_dir', opt.checkpoints_dir)
local netG = util.load(paths.concat(opt.checkpoints_dir, opt.netG_name .. '.t7'), opt)
--netG:evaluate()
print(netG)

-- web
local app = require 'waffle'

app.get('/', function(req, res)
   res.send(html { body { form {
      method = 'POST',
      enctype = 'multipart/form-data',
      p { input {
         type = 'file',
         name = 'file'
      }},
      p { input {
         type = 'submit',
         'Upload'
      }}
   }}})
end)

app.post('/', function(req, res)
  local input = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
  local target = torch.FloatTensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

  local imageTempPath = '/tmp/tmp.jpg'
  local options = {
    path = imageTempPath
  }

  local function cb(err)
    -- Get Data
    local data_curr, filepaths_curr

    -- Only for colorization
    input_lab = loadImageChannel(imageTempPath)
    input_size = input_lab:size()
    data_curr = torch.Tensor(1, input_size[1], input_size[2], input_size[3])
    data_curr[1]:copy(input_lab)
    filepaths_curr = {}
    filepaths_curr[1] = imageTempPath

    filepaths_curr = util.basename_batch(filepaths_curr)
    -- print('filepaths_curr: ', filepaths_curr)

    input = data_curr[{ {}, idx_A, {}, {} }]
    target = data_curr[{ {}, idx_B, {}, {} }]

    if opt.gpu > 0 then
        input = input:cuda()
    end

    if opt.preprocess == 'colorization' then
       local output_AB = netG:forward(input):float()
       local input_L = input:float() 
       output = util.deprocessLAB_batch(input_L, output_AB)
       local target_AB = target:float()
       target = util.deprocessLAB_batch(input_L, target_AB)
       input = util.deprocessL_batch(input_L)
    else 
        output = util.deprocess_batch(netG:forward(input))
        input = util.deprocess_batch(input):float()
        output = output:float()
        target = util.deprocess_batch(target):float()
    end
    paths.mkdir(paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase))
    local image_dir = paths.concat(opt.results_dir, opt.netG_name .. '_' .. opt.phase, 'images')
    paths.mkdir(image_dir)
    paths.mkdir(paths.concat(image_dir,'input'))
    paths.mkdir(paths.concat(image_dir,'output'))
    paths.mkdir(paths.concat(image_dir,'target'))
    -- print(input:size())
    -- print(output:size())
    -- print(target:size())
    for i=1, opt.batchSize do
        -- image.save(paths.concat(image_dir,'input',filepaths_curr[i]), image.scale(input[i],input[i]:size(2),input[i]:size(3)/opt.aspect_ratio))
        image.save(paths.concat(image_dir,'output',filepaths_curr[i]), image.scale(output[i],output[i]:size(2),output[i]:size(3)/opt.aspect_ratio))
        -- image.save(paths.concat(image_dir,'target',filepaths_curr[i]), image.scale(target[i],target[i]:size(2),target[i]:size(3)/opt.aspect_ratio))
    end

    res.send('Saved images to: ' .. image_dir)
  end

  req.form.file:save(options, cb)

end)

app.listen()

