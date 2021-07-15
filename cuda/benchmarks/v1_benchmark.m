clear all
close all
clc

format short g

dirname = '/home/luca/Desktop/uni/lab/hpc/hpc-skyline/src/cuda/version1/stdout/';
tests = dir(fullfile(dirname, 'v1-*'));
sizeA = [2 inf];
formatSpec = '%d %lf';

figure
cla
hold on
colors = ["k*-" "m*-" "k*-" "b*-" "y*-" "g*-" "c*-", "r-*"]; 
%colors = "gyr"
for k = 1:length(tests)
    filename = fullfile(dirname, tests(k).name);
    disp(filename)
    f = fopen(filename, "r");
    A = fscanf(f, formatSpec, sizeA)
    
    % numero di blocchi nella griglia
    %gridsizeY = floor((100400 + A(1, :) - 1) ./ A(1, :));
    % numero di thread dell'ultimo blocco che non computano
    %v = mod(gridsizeY .* A(1, :), 100400);
    % numero di thread dell'ultimo blocco che computano
    %t = (A(1, :) - v);
    % numero di thread dell'ultimo warp dell'ultimo blocco
    % NB: avanzano sempre 16 threads, quindi l'ultimo warp è
    % utilizzato al 50%
    %ris = mod(t, 32);
    
    [M, index] = min(A(2, :));
    plot(A(1, :), A(2, :), colors(k), 'LineWidth', 1.5); 
    plot(A(1, index), M, 'ko', 'LineWidth', 1.5);

    
    fclose(f);
end

legend('test7 mrot', 'test7 mrot min', 'test1', 'test1 min', 'test2', 'test2 min', 'test3', 'test3 min', ...
    'test4', 'test4 min', 'test5', 'test5 min', 'test6', 'test6 min', 'test7', 'test7 min')

saveas(gcf, "prova.png");
