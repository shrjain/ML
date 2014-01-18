function [  ] = runCCLorma(rank, contextMethod, dset, contextType)
    if dset == 10
        load('ratings10M.mat')
        load('genres10M.mat')
    else
        load('ratings.mat')
        load('genres.mat')
    end
    load('userMovieLens.mat')
    rating = data(:,5); % ratings are from 1 - 5
    Gender = cell2mat(Gender);
    % Get All the movie ID's
    movieID = data(:,3); % All movies are not present
    
    n_movies = max(movieID);
    moviemapping = 1:n_movies;
    movieID_All = movieID;
    
    % Get the unique user ID's and map them in some way
    % Each user ID in uniq_userID maps to a number between 1 and numel(uniq_userID)
    userID = data(:,1); % All users are present
    n_users = max(userID);
    mapping = 1:n_users;
    userID_All = userID;
    
    % Take a random percentage of the data for testing & rest for training
    % train:test = 8:1
    n_data = numel(userID);
    n_train = round(0.8*n_data); % 80% for training
    n_test = n_data - n_train; % 20% for testing
    rperm = randperm(n_data); % Get a random permutation of the data

    % Now create a sparse matrix with JUST the training part of the data
    userID_Train = userID_All(rperm(1:n_train));
    movieID_Train = movieID_All(rperm(1:n_train));
    M = sparse(userID_Train,movieID_Train,double(rating(rperm(1:n_train))));
    uniquserID_Train = unique(userID_Train);
    uniqmovieID_Train = unique(movieID_Train);
    vect = 1:n_train;
    A = (M ~= 0);
    rateCount = nnz(M);

    % CHOOSE PARAMETERS FOR THE OPTIMIZATION
    r = rank; % rank of the decomposition
    h1 = 0.8; % width of epanechnikov kernel
    h2 = 0.8; % width of epanechnikov kernel
    lambda_U = 0.001; % L2 regularization weight for U
    lambda_V = 0.001; % L2 regularization weight for v
    T = 100; % No of gradient descent iterations
    epsilon = 0.0001; % Convergence criterion for Gradient descent (was 0.0001)
    eta = 0.01; %step size for gradient descent (was 0.01 originally)
    q = 50; % No of anchor points

    % Compute an Incomplete SVD of the observed values for the distance metric
    [U_I,s_I,V_I] = svds(M,r); % Do a rank "r" approximate of the matrix M

    normU_I = sum(U_I.^2,2).^0.5; % Norm of the user singular vectors
    normV_I = sum(V_I.^2,2).^0.5; % Norm of the movie singular vectors
    
    normGenres = sum(genres.^2,2).^.5;
    
    genresAndV = vertcat(V_I',genres')'; % Movie Singular vectors combined with 
    normGenresAndV = sum(genresAndV.^2,2).^.5; % Norm of 
    
    U_k = rand(size(U_I,1),size(U_I,2),q);
    V_k = rand(size(V_I,1),size(V_I,2),q);
    mov_anchor = zeros(1,q);
    user_anchor = zeros(1,q);
    K_user_anchor = zeros(size(U_I,1),q);
    K_mov_anchor = zeros(size(V_I,1),q);

    for k = 1:q
        
        % Choose a user and movie randomly
        while(1)
            user_r = floor(rand(1)*n_users)+1;
            [~,cx] = find(M(user_r,:)); % Get indices of non zero values
            if isempty(cx)
                continue;
            end
            randID = floor(rand(1)*numel(cx))+1;
            mov_r = cx(randID);
            break;
        end


        mov_anchor(k) = mov_r;
        user_anchor(k) = user_r;

        %fprintf('Anchor point no: %d/%d \n',k,q);

        % Compute distance to every other user using epanechnikov kernel &
        % distance as a function of the angle between the rows of U

        % Compute distance to every other movie using epanechnikov kernel &
        % distance as a function of the angle between the rows of V
        
        d_user = 1 - (2/pi)*real(acos( (U_I*U_I(user_r,:)') ./ (normU_I*normU_I(user_r)) ));
        K_user = max((3/4)*(1 - (d_user/h1).^2),0);

        filter1 = 1;
        filter2 = 1;

        d_mov = 1 - (2/pi)*real(acos( (V_I*V_I(mov_r,:)') ./ (normV_I*normV_I(mov_r)) ));
        if contextMethod == 1
            if contextType == 1
                filter1 = Gender == Gender(user_r);
                filter1(filter1 == 0) = 0.9;
            elseif contextType == 2
                filter2 = 6./abs(AgeGroup(user_r) - AgeGroup);
                filter2(filter2 == Inf) = 1;
            elseif contextType == 3
                d_mov = 1 - (2/pi)*real(acos( (genresAndV*genresAndV(mov_r,:)') ./ (normGenresAndV*normGenresAndV(mov_r)) ));
            elseif contextType == 4
                filter1 = Gender == Gender(user_r);
                filter1(filter1 == 0) = 0.9;
                filter2 = 6./abs(AgeGroup(user_r) - AgeGroup);
                filter2(filter2 == Inf) = 1;
                d_mov = 1 - (2/pi)*real(acos( (genresAndV*genresAndV(mov_r,:)') ./ (normGenresAndV*normGenresAndV(mov_r)) ));
            end
        end
        
        K_user = K_user .* filter1 .*filter2;
        d_validuser = mapping(K_user ~= 0);
        %fprintf('No of users covered by this point: %d \n',sum(K_user ~= 0));
        
        K_mov = max((3/4)*(1 - (d_mov/h2).^2),0);
        d_validmovie = moviemapping(K_mov ~= 0);
        %fprintf('No of movies covered by this point: %d \n',sum(K_mov ~= 0));


        U_temp = U_k(:,:,k); % SOME WEIRDNESS HERE. IF I PASS IN THE 3D MATRIX, IT IS NOT GETTING CHANGED PROPERLY
        V_temp = V_k(:,:,k);
        % USE FAST C-CODE TO DO THE LLORMA!!
        iterate_LLORMA(M', U_temp,V_temp,K_user,K_mov);
        U_k(:,:,k) = U_temp;
        V_k(:,:,k) = V_temp;

        if contextMethod == 2
            if contextType == 1
                filter1 = Gender == Gender(user_r);
                filter1(filter1 == 0) = 0.9;
            elseif contextType == 2
                filter2 = 6./abs(AgeGroup(user_r) - AgeGroup);
                filter2(filter2 == Inf) = 1;
            elseif contextType == 3
                d_movie_filter2 = 1 - (2/pi)*real(acos( (genresAndV*genresAndV(mov_r,:)') ./ (normGenresAndV*normGenresAndV(mov_r)) ));
                movie_filter2 = max((3/4)*(1 - (d_movie_filter2/h2).^2),0);
                K_mov = K_mov .* full(movie_filter2);
            elseif contextType == 4
                filter1 = Gender == Gender(user_r);
                filter1(filter1 == 0) = 0.9;
                filter2 = 6./abs(AgeGroup(user_r) - AgeGroup);
                filter2(filter2 == Inf) = 1;
                d_movie_filter2 = 1 - (2/pi)*real(acos( (genresAndV*genresAndV(mov_r,:)') ./ (normGenresAndV*normGenresAndV(mov_r)) ));
                movie_filter2 = max((3/4)*(1 - (d_movie_filter2/h2).^2),0);
                K_mov = K_mov .* full(movie_filter2);
            end
            K_user = K_user .* filter1 .*filter2;
        end
        K_mov_anchor(:,k) = K_mov;
        K_user_anchor(:,k) = K_user;
    end

    % Test the results on test dataset
    userID_Test = userID_All(rperm(n_train+1:end));
    movieID_Test = movieID_All(rperm(n_train+1:end));
    testPred = zeros(size(userID_Test));
    actRatings = double(rating(rperm(n_train+1:end)));

    %Remove ID's that were not present in training data. We assign default
    %rating '3' to them
    pUserMov = ismember(userID_Test,uniquserID_Train) & ismember(movieID_Test,uniqmovieID_Train); % these guys have training data

    % Find the K values for test user & test movies for each anchor point
    K_test = zeros(numel(userID_Test),q);
    K_testUser = zeros(numel(userID_Test),q);
    K_testMov = zeros(numel(userID_Test),q);
    for k = 1:q
        K_test(:,k) = K_user_anchor(userID_Test,k) .* K_mov_anchor(movieID_Test,k);
        K_testUser(:,k) = K_user_anchor(userID_Test,k);
        K_testMov(:,k) = K_mov_anchor(movieID_Test,k);
    end

    %
    %Now compute predictions for each of the test points
    idZ = [];
    ct = 0;
    for j = 1:numel(userID_Test)
        if ~pUserMov(j) % MEANS WE DO NOT HAVE ANY TRAINING DATA FOR EITHER THE USER OR MOVIE
            testPred(j) = 3.0; % default rating
        else
            testPred(j) = sum(reshape(K_test(j,:),1,1,q).*(sum(U_k(userID_Test(j),:,:).*V_k(movieID_Test(j),:,:),2)),3)./sum(K_test(j,:));
            if isnan(testPred(j)) % this means that this point was not within the local region of any of the anchor points (sum(K_test(j,:)) = 0 & dividing by 0 gives NaN)
                testPred(j) = 3.0; % default rating
                idZ = [idZ j];
            elseif testPred(j) > 5
                testPred(j) = 5.0;
            elseif testPred(j) < 1
                testPred(j) = 1.0;
            end
        end
    end
    %}

    RMSE = sqrt(mean((actRatings - testPred).^2));
    MEA = mean(abs(actRatings - testPred));
    fprintf('RMSE for rank: %d and %d anchor points is: %f \n', r, q, RMSE);
    %fprintf('MEA for rank: %d and %d anchor points is: %f \n', r, q, MEA);
    %fprintf('Total no of test points: %d, No of points without any data/no support: %d \n',numel(userID_Test),numel(idZ)+sum(~pUserMov));
end

