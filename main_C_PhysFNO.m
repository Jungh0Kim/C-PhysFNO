% Jungho Kim
% junghokim@sejong.ac.kr
% Kim, J., Yi, S., and Wang, Z. (2026). A composition of simplified physics-based
% model with neural operator for trajectory-level seismic response predictions
% of structural systems. Structural Safety, 119, 102668.
% https://doi.org/10.1016/j.strusafe.2025.102668
% 
clear; close all; tic;
set(0,'DefaultFigureColor',[1 1 1]);
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(0,'defaulttextinterpreter','latex','DefaultAxesFontSize',14);
hei = 10;  wid = hei*1.618;  resolution = '-r600';
CP = [241 40 21; 64	86 161; 215	153	34; 20 160 152; 203 45 111]/255;

%% Load data

% linearized and original responses for Auburn Ravine bridge (Ex 5.2)
load('ARB_gen2000_data.mat')

% Synthetic motion dataset
n_Data = size(GM_gen.ag,1);
GM_sz = zeros(n_Data,1);
dt_sv = zeros(n_Data,1);
for kk=1:n_Data
    GM_sz(kk,1) = size(GM_gen.ag{kk,1}, 2);
    dt_sv(kk,1) = GM_gen.t{kk,1}(10) - GM_gen.t{kk,1}(9);
end
min(GM_sz)      % min_Tau = 3666 
mean(dt_sv)     % dt = 0.01
dt = 0.01;
N_tau = 3001;

%% Set parameters

rng(10)

n_Data = size(GM_gen.ag,1);
Tau = dt*(N_tau-1);
N_DoF = 10;  % 1~10 for bridge piers
g = 9.81;
input = zeros(n_Data, N_tau, N_DoF);
output = zeros(n_Data, N_tau, N_DoF);
for kk=1:n_Data
    for ll=1:N_DoF
        input(kk,:, ll) = D_hist_linear{kk,ll}(:,1:N_tau);  % Input: linearized response
    end
end
for kk=1:n_Data
    for ll=1:N_DoF
        output(kk,:, ll) = D_hist_NL{kk,ll}(:,1:N_tau);  % Output: Original response
    end
end

N_train = 600;
N_vali = 100;
N_test = 200;
gridSize = size(input,2);

%% View training data

n_pp = 3;

figure()
plot(linspace(0,Tau,gridSize), GM_gen.ag{n_pp,1}(:,1:N_tau)./g,'k')
xlabel('$t$, sec','fontsize',15)
ylabel('$a_g(t)$, g','fontsize',15)
xlim([0 Tau])
set(gcf,'unit','centimeters','position',[0 0 0.8*wid 0.6*hei]);

figure()
plot(linspace(0,Tau,gridSize),squeeze(input(n_pp,:,1)),'linewidth', 1.1, 'color','k','linestyle','--'); hold on;
plot(linspace(0,Tau,gridSize),squeeze(output(n_pp,:,1)),'linewidth', 1.3, 'color',CP(2,:),'linestyle','-');
xlabel('$t$, sec','fontsize',15)
ylabel(strcat('$u_1$, m'),'fontsize',15)
xlim([0 Tau])
set(gcf,'unit','centimeters','position',[0 0 0.8*wid 0.6*hei]);

%% Prepare training data for neural operator learning

N_data = size(input,1);
[idx_train2, idx_vali2, idx_test2] = trainingPartitions(N_data, [0.7 0.1 0.2]);
idx_train = idx_train2(1, 1:N_train);
idx_vali = idx_vali2(1, 1:N_vali);
idx_test = idx_test2(1, 1:N_test);

Downsample_size = N_tau;
ds_factor = floor(gridSize./Downsample_size);

x_train = input(idx_train, 1:ds_factor:end,:);
y_train = output(idx_train, 1:ds_factor:end,:);
x_vali = input(idx_vali, 1:ds_factor:end,:);
y_vali = output(idx_vali, 1:ds_factor:end,:);
x_test = input(idx_test, 1:ds_factor:end,:);
y_test = output(idx_test, 1:ds_factor:end,:);

% Input data to have a channel that corresponds to the spatial coordinates
xMin = 0;
xMax = Tau;
xGrid = linspace(xMin, xMax, Downsample_size);

xGrid_Train = repmat(xGrid, [N_train 1 1]);  % [N_train × N_tau × 1]
xGrid_Vali = repmat(xGrid, [N_vali 1 1]);
xGrid_Test = repmat(xGrid, [N_test 1 1]);

x_Train_grid = cat(3, x_train, xGrid_Train);
x_Vali_grid = cat(3, x_vali, xGrid_Vali);
x_Test_grid = cat(3, x_test, xGrid_Test);

% View the sizes of the training and validation sets.
% The first, second, and third dimensions are the batch, spatial, and channel dimensions, respectively.
size(x_Train_grid)     % [Batch × Time × (N_DoF+1)].
size(x_Vali_grid)

% B = Batch Dimension (Number of samples)
% S = Spatial Dimension (Time points)
% C = Channel Dimension (Features at each time) 
% [Batches, Time Steps, (N_DoF + 1)]

%% Neural operator architecture

num_Modes = 40;        % Number of Fourier modes
spatial_width = 128;   % Number of hidden features at every discrete t (spatial width)

layers = [
    % initial BSC (# of channel is N_DoF+1)
    inputLayer([NaN spatial_width (N_DoF+1)], "BSC");
    % inputLayer([NaN Downsample_size (N_DoF+1)], "BSC");

    % "Lifting" layer from (N_DoF+1) to (spatial_width)
    convolution1dLayer(1, spatial_width, Name="fc0")

    % Fourier layers with "spatial_widt" channel
    fourierLayer(spatial_width, num_Modes, Name="fourier1");
    reluLayer
    fourierLayer(spatial_width, num_Modes, Name="fourier2")
    reluLayer
    fourierLayer(spatial_width, num_Modes, Name="fourier3")
    reluLayer
    fourierLayer(spatial_width, num_Modes, Name="fourier4")
    reluLayer
    fourierLayer(spatial_width, num_Modes, Name="fourier5")
    reluLayer  
    fourierLayer(spatial_width, num_Modes, Name="fourier6")
    reluLayer  
    fourierLayer(spatial_width, num_Modes, Name="fourier7")
    reluLayer  
    fourierLayer(spatial_width, num_Modes, Name="fourier8")
    reluLayer 

    % two-stage projection layer from (spatial_width)-> 128 -> N_DoF
    convolution1dLayer(1, 128)
    reluLayer
    convolution1dLayer(1, N_DoF)
    ];

%% Training
% Train using a piecewise learning schedule, with an initial learning rate of 0.001 and a drop factor of 0.5.
% Shuffle the data every epoch.
% Monitor the training progress in a plot and specify the validation data.
% Disable the verbose output.

options = trainingOptions("adam", ...
    InitialLearnRate = 1e-3, ...       
    LearnRateSchedule = piecewiseLearnRate(DropFactor=0.5), ...
    MaxEpochs = 100, ...            % Number of complete passes over the training dataset
    MiniBatchSize = 20, ...         % Number of samples per mini-batch update
    Shuffle = "every-epoch", ...
    InputDataFormats = "BSC", ...
    Plots = "training-progress", ...
    ValidationData = {x_Vali_grid,y_vali}, ...
    Verbose = false);

% By default, the trainnet function uses a GPU if one is available
net = trainnet(x_Train_grid, y_train, layers, @relativeL2Loss_all, options);
% analyzeNetwork(net)

%% Test Network

y_pred = minibatchpredict(net, x_Test_grid);  % [N_test × N_tau × N_DoF]

% Per-DoF RMSE
abs_error = abs(y_pred - y_test);          % [N_test × N_tau × N_DoF]
rmse = sqrt(mean(abs_error.^2, [1 2]));    % RMSE per over batch and time
rmse = reshape(rmse, N_DoF, 1);
Relative_L2 = rmse./reshape(sqrt(mean(y_test.^2, [1 2])), N_DoF, 1);

disp('RMSE and relativeL2 for each DoF:')
disp(rmse')
disp(Relative_L2')

%% Predict and visualize new data

N_pp = 10;    % Index of the test sample to visualize

input_new = input(idx_test(N_pp),:,:);
output_new = output(idx_test(N_pp),:,:);

gridsize_full = size(input_new,2);
xgrid_full = linspace(xMin, xMax, gridsize_full);
xgrid_full = repmat(xgrid_full, [size(input_new,1) 1 1]);
x_New_grid = cat(3, input_new, xgrid_full);

% Make predictions using the minibatchpredict function
y_pred_new = minibatchpredict(net, x_New_grid);
abs_error_new = abs(y_pred_new - output_new);     

% Visualize the predictions in a plot.
figure()
plot(0:dt:dt*(N_tau-1), GM_gen.ag{idx_test(N_pp),1}(1:N_tau)./g,'linewidth', 1.2,'color','k')
xlabel('$t$, sec','fontsize',15)
ylabel('$a_g(t)$, g','fontsize',15)
xlim([0 Tau])
set(gcf,'unit','centimeters','position',[0 0 0.9*wid 0.6*hei]);
% tightfig;

figure()
tiledlayout(ceil(N_DoF/2),1)
for i=1:ceil(N_DoF/2)
    nexttile
    plot(xgrid_full, squeeze(output_new(:,:,2*i-1)),'linewidth', 1.4, 'linestyle', '-', 'color',CP(2,:)); hold on;
    plot(xgrid_full, squeeze(y_pred_new(:,:,2*i-1)),'linewidth', 1.4, 'linestyle', '--', 'color',CP(1,:));
    xlabel('$t$, sec','fontsize',15)
    ylabel(strcat('$u_{',num2str(2*i-1),'}$, m'),'fontsize',15)
    xlim([0 Tau])
    if i==5
        legend('Ground truth','Prediction')
    end
end
set(gcf,'unit','centimeters','position',[0 0 0.8*wid 2.5*hei]);
% tightfig;

figure()
tiledlayout(ceil(N_DoF/2),1)
for i=1:ceil(N_DoF/2)
    nexttile
    plot(xgrid_full, squeeze(abs_error_new(:,:,2*i-1)),'linewidth', 1.0, 'linestyle', '-', 'color','k'); hold on;
    xlabel('$t$, sec','fontsize',15)
    ylabel(strcat('$\epsilon_{',num2str(2*i-1),'}$, m'),'fontsize',15)
    xlim([0 Tau])
    ax1 = gca;   
    ax1.YMinorGrid = 'on';
    ax1.MinorGridLineStyle = '-';
    ax1.MinorGridAlpha = 0.1;
    ax1.MinorGridColor = [0.5 0.5 0.5];
    % ax1.YAxis.Exponent = 2;
    ax1.YAxis.Scale ="log";
    ylim([1e-6 1e-2])
    hold off
end
set(gcf,'unit','centimeters','position',[0 0 0.8*wid 2.5*hei]);
% tightfig;

%% Linear regression-based refinement of C-PhysFNO (Section 4)

y_train_pred = minibatchpredict(net, x_Train_grid);  
gray = [162, 160, 165]/255;

DoF = 10;
y_train0 = y_train(:,:,DoF);
y_train_pred0 = y_train_pred(:,:,DoF);
x_train0 = x_train(:,:,DoF);

Nsamp = 10^3;    % randomly select Nsamp points for training the linear regression
idx = randperm(N_train*N_tau, Nsamp); 
y_train1 = y_train0(idx)';
x_train1 = x_train0(idx)';
y_train_pred1 = y_train_pred0(idx)';

% linear regression
Nsamp = numel(y_train1);
S = ones(Nsamp,3);
S(:,2) = y_train_pred1;
S(:,3) = x_train1;

w = (inv(S'*S))*S'*y_train1;
mu = S*w;
sigma = sqrt((y_train1-mu)'*(y_train1-mu)/Nsamp);
Sp = ones(N_test*N_tau,3);
Sp(:,2) = reshape(y_pred(:,:,DoF)',[N_tau*N_test,1]);
Sp(:,3) = reshape(x_Test_grid(:,:,DoF)',[N_tau*N_test,1]);
mup = Sp*w;

N_pp = 10;

figure()
p1 = plot(0:dt:Tau,y_test(N_pp,:,DoF),'Color','black','LineWidth',1);
hold on
p2 = plot(0:dt:Tau, mup(N_tau*(N_pp-1)+1:N_tau*N_pp),'--','Color',CP(1,:),'LineWidth',1);
p3 = plot(0:dt:Tau, mup(N_tau*(N_pp-1)+1:N_tau*N_pp)+sigma,'--','linewidth',.5,'color',gray);
plot(0:dt:Tau, mup(N_tau*(N_pp-1)+1:N_tau*N_pp)-sigma,'--','linewidth',.5,'color',gray);
patch([0:dt:Tau,fliplr(0:dt:Tau)],[mup(N_tau*(N_pp-1)+1:N_tau*N_pp)'+sigma',fliplr(mup(N_tau*(N_pp-1)+1:N_tau*N_pp)'-sigma')],gray,'Edgecolor','none','FaceAlpha',0.2)
legend([p1,p2,p3],'Ground truth','Mean prediction','Prediction interval','Location','northwest');
title('Refined C-PhysFNO')
xlabel('$t$, sec','fontsize',16)
ylabel(strcat('$u_{',num2str(DoF),'}$, m'),'fontsize',16.5)
set(gcf, 'PaperPosition', [0 0 0.9*wid 0.9*hei]);
% tightfig;

%% Prediction error
rmse_CPhysFNO = sqrt(mean(abs(Sp(:,2) - reshape(y_test(:,:,DoF)',[N_tau*N_test,1])).^2));
rmse_Refined = sqrt(mean(abs(mup - reshape(y_test(:,:,DoF)',[N_tau*N_test,1])).^2));
Rel_CPhysFNO = rmse_CPhysFNO./sqrt(mean(y_test(:,:,DoF).^2, [1 2]));
Rel_Refined = rmse_Refined./sqrt(mean(y_test(:,:,DoF).^2, [1 2]));

toc;
