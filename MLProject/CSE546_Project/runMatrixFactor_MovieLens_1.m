clear all;
close all;
data = dlmread('ml-1m/ratings.dat');

rating = data(:,5); % ratings are from 1 - 5

% Get All the movie ID's
movieID = data(:,3); % All movies are not present

n_movies = max(movieID);
moviemapping = 1:n_movies;
movieID_All = movieID;
%}

% Get the unique user ID's and map them in some way
% Each user ID in uniq_userID maps to a number between 1 and numel(uniq_userID)
userID = data(:,1); % All users are present

%
n_users = max(userID);
mapping = 1:n_users;
userID_All = userID;
%}

% Take a random percentage of the data for testing & rest for training
% train:test = 9:1
n_data = numel(userID);
n_train = round(0.9*n_data); % 90% for training
n_test = n_data - n_train; % 10% for testing
rperm = randperm(n_data); % Get a random permutation of the data

% Now create a sparse matrix with JUST the training part of the data
userID_Train = userID_All(rperm(1:n_train));
movieID_Train = movieID_All(rperm(1:n_train));
M = sparse(userID_Train,movieID_Train,double(rating(rperm(1:n_train))));
gobalAverageRating = mean(rating(rperm(1:n_train)));
%{
Mx = full(M);
Mx(Mx ==0) = gobalAverageRating; % DEFAULT RATING
M = Mx;
%}
uniquserID_Train = unique(userID_Train);
uniqmovieID_Train = unique(movieID_Train);
vect = 1:n_train;
A = (M ~= 0);

% CHOOSE PARAMETERS FOR THE OPTIMIZATION
r = 5; % rank of the decomposition
h1 = 0.8; % width of epanechnikov kernel
h2 = 0.8; % width of epanechnikov kernel
lambda_U = 0.001; % L2 regularization weight for U
lambda_V = 0.001; % L2 regularization weight for v
T = 100; % No of gradient descent iterations
epsilon = 10; % Convergence criterion for Gradient descent
eta = 0.0001; %step size for gradient descent (was 0.01 originally)
q = 50; % No of anchor points

% Compute an Incomplete SVD of the observed values for the distance metric
[U_I,s_I,V_I] = svds(M,r); % Do a rank "r" approximate of the matrix M
U_I = U_I*s_I;
normU_I = sum(U_I.^2,2).^0.5; % Norm of the user singular vectors
normV_I = sum(V_I.^2,2).^0.5; % Norm of the movie singular vectors


U_k = zeros(size(U_I,1),size(U_I,2),q);
V_k = zeros(size(V_I,1),size(V_I,2),q);
mov_anchor = zeros(1,q);
user_anchor = zeros(1,q);
K_user_anchor = zeros(size(U_I,1),q);
K_mov_anchor = zeros(size(V_I,1),q);

for k = 1:q
    randID = round(rand(1)*n_train); % Get a random index from the training dataset  
    mov_r = movieID_Train(randID); % Column index of M
    user_r = userID_Train(randID); % Row index of M
    
    mov_anchor(k) = mov_r;
    user_anchor(k) = user_r;
    
    % Compute distance to every other user using epanechnikov kernel &
    % distance as the angle between the rows of U
    % (arccos(dot(U_i,U_j)/||U_i||||U_j||))
    % d_user = real(acos( (U_I*U_I(user_r,:)') ./ (normU_I*normU_I(user_r)) ));
    d_user = real(1 - acos( (U_I*U_I(user_r,:)') ./ (normU_I*normU_I(user_r)) ) * (2/pi));
    %d_validuser = mapping(d_user < h1);
    %K_user = (1 - d_user.^2).*(d_user < h1);
    K_user = max(3.0/4.0 * (1 - (d_user ./ h1) .^ 2), 0);
    d_validuser = mapping(K_user ~= 0);
    fprintf('No of users covered by this point: %d \n',sum(d_user < h1));
    %K_user = K_user ./ norm(K_user);
    %K_user = K_user ./ sum(K_user);
    K_user_anchor(:,k) = K_user;
    
    % Compute distance to every other movie using epanechnikov kernel &
    % distance as the angle between the rows of V
    % (arccos(dot(V_i,V_j)/||V_i||||V_j||))
    % d_mov = real(acos( (V_I*V_I(mov_r,:)') ./ (normV_I*normV_I(mov_r)) ));
    d_mov = real(1 - acos( (V_I*V_I(mov_r,:)') ./ (normV_I*normV_I(mov_r)) ) * (2/pi) );
    %d_validmovie = moviemapping(d_mov < h2);
    %K_mov = (1 - d_mov.^2).*(d_mov < h2);
    K_mov = max(3.0/4.0 * (1 - (d_mov ./ h2) .^ 2), 0);
    d_validmovie = moviemapping(K_mov ~= 0);
    fprintf('No of movies covered by this point: %d \n',sum(d_mov < h2));
    %K_mov = K_mov ./ norm(K_mov);
    %K_mov = K_mov ./ sum(K_mov);
    K_mov_anchor(:,k) = K_mov;


    % Get the full user x movies matrix
    %K_full = K_user * K_mov';
    
    %
    kernel_union = ismember(userID_Train,d_validuser) & ismember(movieID_Train,d_validmovie);
    userID_All_filt = userID_Train(kernel_union); % users that are similar to the current anchor point
    movieID_All_filt = movieID_Train(kernel_union); % movies that are similar to the current anchor point
    valid_indices = sub2ind(size(M),userID_All_filt,movieID_All_filt);
    %}
        
    %U_k(:,:,k) = U_I*s_I; 
    %V_k(:,:,k) = V_I;
    U_k(:,:,k) = rand(size(U_I));
    V_k(:,:,k) = rand(size(V_I));
    
    %U_k(:,:,k) = zeros(size(U_I));
    %V_k(:,:,k) = zeros(size(V_I));
    
    loss_function_prev = inf;
    Mprime = bsxfun(@times,bsxfun(@times, M, K_user),K_mov');
    
    sumTerm = sum(bsxfun(@times,K_user(userID_All_filt),U_k(userID_All_filt,:,k)).*(bsxfun(@times,K_mov(movieID_All_filt),V_k(movieID_All_filt,:,k))),2) - Mprime(valid_indices);
    tempmat = zeros(numel(userID_Train),1);
    
    % Now do gradient descent
    iteration = 0;
    not_converged = true;
    while not_converged
        iteration = iteration + 1;

        tempmat(kernel_union) = sumTerm;
        tempmat(~kernel_union) = 0;
        UVTransMinusM = sparse(userID_Train,movieID_Train,tempmat);
        
        U_k(:,:,k) = U_k(:,:,k) - eta*2*(lambda_U*U_k(:,:,k) + UVTransMinusM*V_k(:,:,k)); % AM I MISSING USING K_MOV HERE?
        V_k(:,:,k) = V_k(:,:,k) - eta*2*(lambda_V*V_k(:,:,k) + (U_k(:,:,k)'*UVTransMinusM)'); % AM I MISSING USING K_USER HERE?
          
        sumTerm = sum(bsxfun(@times,K_user(userID_All_filt),U_k(userID_All_filt,:,k)).*(bsxfun(@times,K_mov(movieID_All_filt),V_k(movieID_All_filt,:,k))),2) - Mprime(valid_indices);
        loss_function_new = lambda_U * norm(U_k(:,:,k), 'fro') + lambda_V * norm(V_k(:,:,k), 'fro') + sum(sumTerm.^2);

        if abs(loss_function_prev - loss_function_new) < epsilon
            fprintf('converged\n');
            not_converged = false;
            break;
        else 
            fprintf('Loss function for iteration %d of point %d is : %f \n',iteration,k,loss_function_new);
            loss_function_prev = loss_function_new;
        end
    end
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

%Now compute predictions for each of the test points
idZ = [];
ct = 0;
gt5 = 0;
lt1 = 0;
for j = 1:numel(userID_Test)
    if ~pUserMov(j) % MEANS WE DO NOT HAVE ANY TRAINING DATA FOR EITHER THE USER OR MOVIE
        testPred(j) = 3.0; % default rating
    else
        testPred(j) = sum(reshape(K_test(j,:),1,1,q).*(sum(U_k(userID_Test(j),:,:).*V_k(movieID_Test(j),:,:),2)),3)./sum(K_test(j,:));
        if isnan(testPred(j)) % this means that this point was not within the local region of any of the anchor points (sum(K_test(j,:)) = 0 & dividing by 0 gives NaN)
            testPred(j) = 3.0; % default rating
            idZ = [idZ j];
        end
    end
    if testPred(j) < 1
        lt1 = lt1 + 1;
        testPred(j) = 1;
    end
    if testPred(j) > 5
        gt5 = gt5 + 1;
        testPred(j) = 5;
    end
end
%}
fprintf('Greater than 5 prediction: %d, Less than 1 predictions: %d \n', gt5, lt1);
RMSE = sqrt(mean((actRatings - testPred).^2));
fprintf('RMSE for rank: %d and %d anchor points is: %f \n', r, q, RMSE);
fprintf('Total no of test points: %d, No of points without any data/no support: %d \n',numel(userID_Test),numel(idZ)+sum(~pUserMov));

