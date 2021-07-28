clear all
close all
clc

dirname = '/home/luca/Desktop/uni/lab/hpc/hpc-skyline/cuda/v1_stdout';
tests = dir(fullfile(dirname, 'v1-*'));
sizeA = [2 inf];
formatSpec = '%d %lf';

figure
hold on
colors = ["r-*" "m*-" "k*-" "b*-" "y*-" "g*-" "c*-"]; 
for k = 1:length(tests)
    filename = fullfile(dirname, tests(k).name);
    f = fopen(filename, "r");
    A = fscanf(f, formatSpec, sizeA);
    
    [M, index] = min(A(2, :));
    plot(A(1, :), A(2, :), colors(k), ... 
        A(1, index), M, 'ko', 'LineWidth', 1.5);
    
    fclose(f);
end

legend('test1', 'test1 min', 'test2', 'test2 min', 'test3', 'test3 min', ...
    'test4', 'test4 min', 'test5', 'test5 min', 'test6', 'test6 min', 'test7', 'test7 min')
