allGenres = containers.Map;
allGenres('Action') = 1;
allGenres('Adventure') = 2;
allGenres('Animation') = 3;
allGenres('Children') = 4;
allGenres('Comedy') = 5;
allGenres('Crime') = 6;
allGenres('Documentary') = 7;
allGenres('Drama') = 8;
allGenres('Fantasy') = 9;
allGenres('Film-Noir') = 10;
allGenres('Horror') = 11;
allGenres('Musical') = 12;
allGenres('Mystery') = 13;
allGenres('Romance') = 14;
allGenres('Sci-Fi') = 15;
allGenres('Thriller') = 16;
allGenres('War') = 17;
allGenres('Western') = 18;
allGenres('IMAX') = 19;
allGenres('(no genres listed)') = 20;

genres = zeros(n_movies,size(values(allGenres)',1));
count = 1;
for i = 1:n_movies
    if find(mId == i)
        if strcmp(G1{count},'')
        else
            genres(i,allGenres(G1{count})) = 1;
        end
        if strcmp(G2{count},'')
        else
            genres(i,allGenres(G2{count})) = 1;
        end
        if strcmp(G3{count},'')
        else
            genres(i,allGenres(G3{count})) = 1;
        end
        if strcmp(G4{count},'')
        else
            genres(i,allGenres(G4{count})) = 1;
        end
        if strcmp(G5{count},'')
        else
            genres(i,allGenres(G5{count})) = 1;
        end
        if strcmp(G6{count},'')
        else
            genres(i,allGenres(G6{count})) = 1;
        end
        count = count + 1;
    else
        genres(i,:) = 1;
    end
end