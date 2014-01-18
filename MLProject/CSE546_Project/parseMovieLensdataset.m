clear all;
close all;

mData = importdata('ml-1m/movies.dat');
uData = importdata('ml-1m/users.dat');
rData = importdata('ml-1m/ratings.dat');


% Create empty struct to store data
n_movies = numel([W.isdir]) - sum([W.isdir]);
n_ToUse = 100;
movieID = zeros(n_ToUse,1,'uint16'); % Movie Id is at most 18000, so we can use uint16 to save memory
userID = [];
rating = [];
numRatingsPerMov = []; % Number of ratings per movie

%movieData = cell(n_ToUse,1);

%profile on;
ndir = 2;
for k = 1+ndir:n_ToUse+ndir %numel(W) % 1 and 2 are . and ..
    
    fprintf(' %d \n',k-ndir);
    D = importdata(strcat(tPath,W(k).name));
    
    %Get the movie id
    temp = regexpi(D{1},'[:]','split');
    %movieID(k-ndir) = str2doubleq(temp{1});%str2num(temp{1});
    movieID(k-ndir) = str2doubleq(temp{1});% Movie Id is at most 18000, so we can use uint16 for it
    
    %
    % Get the users, ratings and dates (Vectorized & low memory)
    S = regexpi(D(2:end),'[,]','split');
    Items = [S{:}];
    
    % STORE AS A SINGLE VECTOR - ADD AN EXTRA VECTOR THAT TELLS HOW MANY
    % RATINGS WE HAVE PER MOVIE
    userID = [userID uint32(str2doubleq(Items(1:3:end)))];
    rating = [rating (uint8(char(Items(2:3:end))) - uint8(48))'];
    numRatingsPerMov = [numRatingsPerMov uint32(numel(D(2:end)))];
    
    % STORE ITEMS AS A CELL ARRAY - TAKES UP MORE MEMORY!
    %{
    % User ID - a 32 bit int
    movieData{k-ndir}.userId = uint32(str2doubleq(Items(1:3:end))); % first terms are user ID's
    
    % Rating is always on a 1 to 5 scale
    movieData{k-ndir}.rating = uint8(char(Items(2:3:end))) - uint8(48); %  we convert to int8 to sace
    %}
    
    %{
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

%save('nf_parsedTrainData_1000N.mat','-v7.3','movieID','movieData');
%save('nf_parsedTrainData_1000N.mat','movieID','userID','rating','numRatingsPerMov');
save(sprintf('nf_parsedTrainData_%d.mat',n_ToUse),'movieID','userID','rating','numRatingsPerMov');

