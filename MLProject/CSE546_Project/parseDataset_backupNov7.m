clear all;
close all;

tPath = 'download/training_set/';
W = dir(tPath);

% Create empty struct to store data
%tData = struct([]);
n_movies = numel([W.isdir]) - sum([W.isdir]);
movieID = zeros(n_movies,1);
movieData = cell(n_movies,1);

%profile on;
ndir = 2;
for k = 1+ndir:1000+ndir %numel(W) % 1 and 2 are . and ..
    fprintf(' %d \n',k-ndir);
    
    D = importdata(strcat(tPath,W(k).name));
    
    %Get the movie id
    temp = regexpi(D{1},'[:]','split');
    movieID(k-ndir) = str2doubleq(temp{1});%str2num(temp{1});
    
    %{
    if movieId(k) - str2doubleq(temp{1}) ~= 0 
        fprintf(' not the same \n');
    end
    %}
    
    %
    % Get the users, ratings and dates (Vectorized)
    S = regexpi(D(2:end),'[,]','split');
    Items = [S{:}];
    movieData{k-ndir}.userId = str2doubleq(Items(1:3:end)); % first terms are user ID's
    movieData{k-ndir}.rating = str2doubleq(Items(2:3:end)); % second term is rating
    movieData{k-ndir}.date = Items(3:3:end); % third is date
    %}
    
    %{
    % Get the users, ratings and dates
    movieData{k}.userId = zeros(numel(D)-1,1);
    movieData{k}.rating = zeros(numel(D)-1,1);
    movieData{k}.date = cell(numel(D)-1,1);
    for j = 2:numel(D)
        temp = regexpi(D{j},'[,]','split');
        movieData{k}.userId(j-1) = str2num(temp{1});
        if movieData{k}.userId(j-1) - str2doubleq(temp{1}) ~= 0 
            fprintf(' not the same \n');
        end
        movieData{k}.rating(j-1) = str2num(temp{2});
        if movieData{k}.rating(j-1) - str2doubleq(temp{2}) ~= 0 
            fprintf(' not the same \n');
        end
        movieData{k}.date{j-1} = temp{3};
    end
    %}
end
%profile viewer;

save('nf_parsedTrainData_1000.mat','-v7.3','movieID','movieData');
