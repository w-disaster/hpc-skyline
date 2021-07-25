clear all
close all
clc

format short g

dirname = '/home/luca/Desktop/uni/lab/hpc/hpc-skyline/src/openmp/speedup/';
formatSpec = '%lf';

figure
cla
hold on
colors = ["k*-" "m*-" "r*-" "b*-" "y*-" "g*-" "c*-", "k-*"]; 

% OpenMP speedup
proc = [1:1:12];
A = [];
for k = 1:7
    for i = 1:12
        stringfile =  strcat('test', int2str(k), "*_", int2str(i), ".txt");
        dir1 = dir(fullfile(dirname,  stringfile));
            
        filename = fullfile(dirname, dir1(1).name);
        %disp(filename)
        f = fopen(filename, "r");
        A(k, i) = fscanf(f, formatSpec);
        fclose(f);
    end
    semilogy(proc, A(k, 1) ./ A(k, :), colors(k), 'LineWidth', 1.5);
end

title('OpenMP speedup');
xlabel('P = N. Processors');
ylabel('T_parallel(1) / T_parallel(P)');
legend("test1", "test2", "test3", "test4", "test5", "test6", "test7");

% read test 8 and 9 of v3
for i = 8:9
    dir1 = dir(fullfile(dirname, strcat('test', int2str(i), '*')));
    filename = fullfile(dirname, dir1(1).name);
    f = fopen(filename, 'r');
    A(i, 12) = fscanf(f, formatSpec);
    fclose(f);
end

% OpenMP v1 vs v2
% read all tests of v2
dirname = '/home/luca/Desktop/uni/lab/hpc/hpc-skyline/src/openmp/v2-times/';
for k = 1:9
    dir1 = dir(fullfile(dirname, strcat('test', int2str(k), "*_", "12.txt")));

    filename = fullfile(dirname, dir1(1).name);
    f = fopen(filename, "r");
    B(k) = fscanf(f, formatSpec);
    fclose(f);
end

% plot
xx = (1:1:9);
figure
plot(xx, A(:, 12), 'r-*', xx, B, 'b-*', 'LineWidth', 1.5); 
xlabel('Test number');
ylabel('Execution time (s)');
title('Old vs final version');
legend('final version', 'old version');

figure
plot(xx(1:7), A(1:7, 1)' ./ A(1:7, 12)', 'r-*', ...
    xx(1:7), A(1:7, 1)' ./ B(1:7), 'b-*', 'LineWidth', 1.5);
legend('Final version', 'Old version');
xlabel('Test number');
ylabel('Speedup');
title('Old version vs final version speedup');

% CUDA: coalesced memory access test
dirname = '/home/luca/Desktop/uni/lab/hpc/hpc-skyline/src/cuda/version1/coalesced/';
xx = (1:1:9);
figure
hold on
for k = 1:4
    for i = 1:9
        dir1 = dir(fullfile(dirname, strcat('v', int2str(k), "_test", int2str(i), "*")));
        filename = fullfile(dirname, dir1(1).name);
        f = fopen(filename, "r");
        C(k, i) = fscanf(f, formatSpec);
        fclose(f);
    end
    plot(xx, C(k, :), colors(k), 'LineWidth', 1.5);
end

xlabel('Test number');
ylabel('Execution time (s)');
legend('Local memory + global not coalesced', 'Global not coalesced + global not coalsced', ...
    'Local memory + global coalesced', 'Global coalesced + global coalesced');

% CUDA speedup
dirname = '/home/luca/Desktop/uni/lab/hpc/hpc-skyline/src/cuda/version1/cuda_speedup/';
for i = 1:7
        dir1 = dir(fullfile(dirname, strcat("test", int2str(i), "*")));

        filename = fullfile(dirname, dir1(1).name);
        f = fopen(filename, "r");
        D(i) = fscanf(f, formatSpec);
        fclose(f);
end

xx = (1:1:7);
Y = A(1:7, 12)';
figure
plot(xx, Y./ D, 'b-*', 'LineWidth', 1.5);
xlabel('Test number');
ylabel('S = T\_openmp(12) / T\_cuda');
title('CUDA speedup');
