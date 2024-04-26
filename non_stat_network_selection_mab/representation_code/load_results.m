%% LOAD FILE
clear;
fclose('all');
results_file = fopen('../results/results_muMAB','r');
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

results = zeros(length(strategies), NSteps);

line = fgets(results_file);
index = 1;
while ~strcmp(line, "FINISH")
    if startsWith(line, "Distribution: ") || startsWith(line, "nu: ") || startsWith(line, "Difficulty: ") || startsWith(line, "Strategy: ")
        line = fgets(results_file);
        continue;
    end
    res = strsplit(line);
    for i = 1:(length(res)-1)
        results(index, i) = str2double(res(i));
    end
    index = index + 1;
    line = fgets(results_file);
end
fclose(results_file);
disp("Reading done");
%% PLOT
x_axis = 1:NSteps;
index = 1;
 for strat=1:length(strategies)
     plot(x_axis,results(index,:), 'LineWidth', 2.5)
     hold on;
     index = index + 1;
end
xlabel("Steps");
ylabel("Total Regret");
%title("Evolution of regret in Scenario 1");
legend(strategies)
grid on;
hold off;