%% LOAD FILE
clear;
fclose('all');
results_file = fopen('../results/time_muMAB','r');
line = fgetl(results_file);

m = 0;
NRuns = 0;
NSteps = 0;
strategies = [];
distributions = [];
difficulties = [];
nu = [];

while ~strcmp(line, "START")
    if startsWith(line, "m: ")
        tmp = strsplit(line);
        m = str2double(tmp(2));
    elseif startsWith(line, "NRuns: ")
        tmp = strsplit(line);
        NRuns = str2double(tmp(2));
    elseif startsWith(line, "NSteps: ")
        tmp = strsplit(line);
        NSteps = str2double(tmp(2));
    elseif startsWith(line, "Strategy: ")
        tmp = strsplit(line);
        for i = 2:length(tmp)
            strategies = [strategies, tmp(i)];
        end
    elseif startsWith(line, "Distribution: ")
        tmp = strsplit(line);
        for i = 2:length(tmp)
            distributions = [distributions, tmp(i)];
        end
    elseif startsWith(line, "Difficulty: ")
        tmp = strsplit(line);
        for i = 2:length(tmp)
            difficulties = [difficulties, tmp(i)];
        end
    elseif startsWith(line, "nu: ")
        tmp = strsplit(line);
        for i = 2:length(tmp)
            nu = [nu, tmp(i)];
        end
    end
    line = fgetl(results_file);
end

results = zeros(1, length(strategies)*length(difficulties)*length(nu)*length(distributions));

line = fgets(results_file);
index = 1;
while ~strcmp(line, "FINISH")
    if startsWith(line, "Distribution: ") || startsWith(line, "nu: ") || startsWith(line, "Difficulty: ") || startsWith(line, "Strategy: ")
        line = fgets(results_file);
        continue;
    end
    res = strsplit(line);
    for i = 1:(length(res)-1)
        results(i, index) = str2double(res(i));
    end
    index = index + 1;
    line = fgets(results_file);
end
fclose(results_file);
disp("Reading done");
%% PLOT
index = 1;
c = categorical(strategies);
figure;
tmp = results(index:index+length(strategies)-1);
bar(c,tmp);
hold on;
index= index+length(strategies);
 xlabel("Strategy");
 ylabel("Average Exec Time (s)");
 hold off;