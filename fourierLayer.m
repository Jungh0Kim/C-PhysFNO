% Define the neural network architecture. The network consists of multiple Fourier layers and convolution layers.
% Define a function that creates a network layer that operates in the frequency domain.
% The layer processes the input by applying convolutions and spectral convolutions and sums the outputs.

% The input is connected to both a spectral convolution layer and a convolution layer.
% The outputs of the two convolution layers are connected to an addition layer.
% The output of the FNO layer is the output of the addition layer.
% A spectral convolutional layer applies convolutions in the frequency domain and is
% particularly useful for solving PDEs as it allows the network to learn complex spatial dependencies.
% To apply the spectral convolution operation, the Fourier layer uses the custom layer spectralConvolution1dLayer.

function layer = fourierLayer(spatialWidth,numModes,args)

arguments
    spatialWidth
    numModes
    args.Name = ""
end
name = args.Name;

net = dlnetwork;

layers = [
    identityLayer(Name="in")
    spectralConvolution1dLayer(spatialWidth, numModes, Name="specConv")
    additionLayer(2,Name="add")];
net = addLayers(net,layers);

layer = convolution1dLayer(1, spatialWidth, Name="fc");
net = addLayers(net,layer);

net = connectLayers(net,"in","fc");
net = connectLayers(net,"fc","add/in2");

layer = networkLayer(net, Name=name);

end % function end