% Compute the relative L2 error between prediction and ground truth

function loss = relativeL2Loss(Y, T)

Y = stripdims(Y, "BSC");
T = stripdims(T, "BSC");

p = vecnorm(T-Y, 2, 2);
q = vecnorm(T, 2, 2);

loss = sum(p./q, 1);

end
