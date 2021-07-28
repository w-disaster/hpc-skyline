clear all
close all
clc

format short g

dirname = '/home/luca/Desktop/hpc-skyline/src/openmp/speedup/';
formatSpec = '%lf';

figure
cla
hold on
colors = ["k*-" "m*-" "r*-" "b*-" "y*-" "g*-" "c*-", "k-*"]; 

input_size = 8 * [100000 * 3, 100000 * 4, 100020 * 10, 100009 * 8, 100000 * 20, ...
    100100 * 50, 100400 * 200, 200000 * 200, 500000 * 200]
for i = 1 : 7
    label(i) = strcat("Input size: ", int2str(input_size(i)), ' bytes');
end

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
xlabel('# Threads');
ylabel('Speedup');
legend(label(1), label(2), label(3), label(4), label(5), ...
    label(6), label(7));

figure
hold on
for k = 1:7
    plot(proc, A(k, :), colors(k), 'LineWidth', 1.5);
%    semilogy(proc, A(k, 1) ./ (A(k, :) .* proc), colors(k), 'LineWidth', 1.5);
end
xlabel('# Threads');
ylabel('Execution time (s)');
legend(label(1), label(2), label(3), label(4), label(5), ...
    label(6), label(7));
title('Execution times');

figure
hold on
for k = 1:7
    semilogy(proc, A(k, 1) ./ (A(k, :) .* proc), colors(k), 'LineWidth', 1.5);
end
xlabel('# Threads');
ylabel('Efficiency');
legend(label(1), label(2), label(3), label(4), label(5), ...
    label(6), label(7));
title('Strong scaling efficiency');

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
dirname = '/home/luca/Desktop/hpc-skyline/src/openmp/v2-times/';
for k = 1:9
    dir1 = dir(fullfile(dirname, strcat('test', int2str(k), "*_", "12.txt")));

    filename = fullfile(dirname, dir1(1).name);
    f = fopen(filename, "r");
    B(k) = fscanf(f, formatSpec);
    fclose(f);
end

% plot
figure
plot(input_size, A(:, 12), 'r-*', ...
    input_size, B, 'b-*', 'LineWidth', 1.5); 
xlabel('Input size (bytes)');
ylabel('Execution time (s)');
title('Final version vs Old version');
legend('Final version', 'Old version');
% 
% figure
% plot(input_size(1:7), A(1:7, 1)' ./ A(1:7, 12)', 'r-*', ...
%     input_size(1:7), A(1:7, 1)' ./ B(1:7), 'b-*', 'LineWidth', 1.5);
% legend('Final version', 'Old version');
% xlabel('Input size (bytes)');
% ylabel('Speedup');
% title('Old version vs final version speedup');

% Weak scaling efficiency
weak_scaling = [1.834744, 3.504249,  5.181387, 6.746453, 8.317036, 10.008745, ...
    11.518777, 13.378024, 14.769079, 16.698485, 18.160204, 20.788687];
figure
plot(proc, weak_scaling(1) ./ weak_scaling, 'b-*', 'LineWidth', 1.5);
legend(strcat("Input size: ", int2str(120000 * 200 * 8), " bytes"))
xlabel('# Threads');
ylabel('Efficiency');
title('Weak scaling efficiency');


% CUDA: coalesced memory access test
dirname = '/home/luca/Desktop/hpc-skyline/src/cuda/version1/coalesced/';
xx = (1:1:9);
figure
hold on
for k = 2:4
    for i = 1:9
        dir1 = dir(fullfile(dirname, strcat('v', int2str(k), "_test", int2str(i), "*")));
        filename = fullfile(dirname, dir1(1).name);
        f = fopen(filename, "r");
        C(k, i) = fscanf(f, formatSpec);
        fclose(f);
    end
    plot(input_size, C(k, :), colors(k), 'LineWidth', 1.5);
end

xlabel('Input size (bytes)');
ylabel('Execution time (s)');
legend('Global not coalesced + global not coalsced', ...
    'Local memory + global coalesced', 'Global coalesced + global coalesced');
title('Memory accesses benchmark');

% CUDA speedup
dirname = '/home/luca/Desktop/hpc-skyline/src/cuda/version1/cuda_speedup/';
for i = 1:7
        dir1 = dir(fullfile(dirname, strcat("test", int2str(i), "*")));

        filename = fullfile(dirname, dir1(1).name);
        f = fopen(filename, "r");
        D(i) = fscanf(f, formatSpec);
        fclose(f);
end
D(8) = 20.080872; D(9) = 63.460696;

Y = A(:, 12)';
figure
plot(input_size, Y./ D, 'g-*', 'LineWidth', 1.5);
xlabel('Input size (bytes)');
ylabel('Speedup = Topenmp(12) / Tcuda');
title('CUDA speedup');
