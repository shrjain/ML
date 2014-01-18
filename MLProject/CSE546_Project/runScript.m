rank = [3 5 7 10];
numberOfIterationsPerRank = 20; % running the algorithm 20 times because the anchor points chosen are random.


%Experiments on MovieLens1M dataset.
fprintf('Running without context for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),0,1,0)
    end
end

fprintf('Running with context method 1 and context type = gender for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),1,1,1)
    end
end

fprintf('Running with context method 1 and context type = age for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),1,1,2)
    end
end

fprintf('Running with context method 1 and context type = genre for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),1,1,3)
    end
end

fprintf('Running with context method 1 and context type = combined for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),1,1,4)
    end
end


fprintf('Running with context method 2 and context type = gender for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),2,1,1)
    end
end

fprintf('Running with context method 2 and context type = age for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),2,1,2)
    end
end

fprintf('Running with context method 2 and context type = genre for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),2,1,3)
    end
end

fprintf('Running with context method 2 and context type = combined for 1M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),2,1,4)
    end
end



%Experiments on MovieLens10M dataset.
fprintf('Running without context for 10M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),0,10,0)
    end
end

fprintf('Running with context method 1 and context type = genre for 10M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),1,10,3)
    end
end

fprintf('Running with context method 2 and context type = age for 10M dataset \n');
for i = 1:size(rank,2)
    for j = 1:numberOfIterationsPerRank
        runCCLorma(rank(i),2,10,3)
    end
end