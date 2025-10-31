
function loss = relativeL2Loss_all(Y, T)
% relativeL2Loss computes the total relative L2 error over all samples and all DoFs.

Y = stripdims(Y, "BSC");
T = stripdims(T, "BSC");

% Compute L2 norms along the time axis
p = vecnorm(T-Y, 2, 2);   % [batch_size × N_DoF]
q = vecnorm(T, 2, 2);     % [batch_size × N_DoF]

% Sum over batch and stories
loss = sum(p, 'all') / sum(q, 'all');
end
