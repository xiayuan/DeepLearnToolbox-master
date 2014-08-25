function net = cnnsetup(net, x, y)
    %assert(~isOctave() || compare_versions(OCTAVE_VERSION, '3.8.0', '>='), ['Octave 3.8.0 or greater is required for CNNs as there is a bug in convolution in previous versions. See http://savannah.gnu.org/bugs/?39314. Your version is ' OCTAVE_VERSION]);
    inputmaps = 1;
    mapsize = size(squeeze(x(:, :, 1)));

    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's')    % pooling layer
            mapsize = mapsize / net.layers{l}.scale;
            assert(all(floor(mapsize)==mapsize), ['Layer ' num2str(l) ' size must be integer. Actual: ' num2str(mapsize)]);
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        if strcmp(net.layers{l}.type, 'c')    % convolution layer
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
            for j = 1 : net.layers{l}.outputmaps  %  output map
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    % 'onum' is the number of labels, that's why it is calculated using size(y, 1). If you have 20 labels so the output of the network will be 20 neurons.
    % 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
    % 'ffb' is the biases of the output neurons.
    % 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);

    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end

%我的注释 2014/03/04
% a[j]是激活值, d[j]是残差,灵敏度, j是l层第j个feature map, feature map即图像与卷积核卷积后所提取的特征图，
% a[j]不是神经网络传统意义的一层的第j个元素，而是第j个feature map所有的元素, 而且已经经过了sigmoid函数，net.layers{1}.a{1} = x,
% net.layer{1}是神经网络第一层，即输入层，net.layers{1}.a{1}即图像的初始值，因为输入层只有一个feature
% map，所以是a{1}
% k{i}{j}是第l层第i个feature map连接第l+1层第j个feature map的卷积核（权值），
% 每一个k{i}{j}都是一个5*5的矩阵，随机初始化rand(5)
% net.layers{l - 1}.a{1}是一个3维矩阵， 最后一维是样本数
% net.layers{l}.d{j} 是一个3维矩阵   比如 d(:,:,1)=[m*n]， d(:,:,2)=[m*n],
% d(:,:,k)=[m*n], 则d为一个 m*n*k 的3维矩阵   length(size(d))=3
