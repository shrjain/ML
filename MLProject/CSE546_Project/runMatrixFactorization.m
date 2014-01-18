clear all;
close all;
%load('nf_parsedTrainData_1000N.mat');
load('nf_parsedTrainData_100.mat');

% Get All the movie ID's
n_movies = numel(movieID);
movieID_All = zeros(size(userID)); % Inflated to take into account the number of votes as well
cSum = [0 cumsum(double(numRatingsPerMov))];
for k = 1:n_movies
    movieID_All(cSum(k)+1:cSum(k+1)) = movieID(k);
end
moviemapping = 1:n_movies;

% Get the unique user ID's and map them in some way
% Each user ID in uniq_userID maps to a number between 1 and numel(uniq_userID)
[uniq_userID,~,IC] = unique(userID); % This also returns sorted (Ascending) output 
n_users = numel(uniq_userID);
mapping = 1:n_users;
userID_All = mapping(IC); % Each user ID in uniq_userID maps to a number between 1 and numel(uniq_userID) (Starting from small to larger)

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
uniquserID_Train = unique(userID_Train);
uniquserID_Test = unique(userID_All(rperm(n_train+1:end))); % if there is an id that doesn't appear in training data, set rating to 3
uniqmovieID_Train = unique(movieID_Train);
uniqmovieID_Test = unique(movieID_All(rperm(n_train+1:end))); % if there is an id that doesn't appear in training data, set rating to 3
vect = 1:n_train;

% CHOOSE PARAMETERS FOR THE OPTIMIZATION
r = 5; % rank of the decomposition
h1 = 0.8; % width of epanechnikov kernel
h2 = 0.8; % width of epanechnikov kernel
lambda_U = 0.001; % L2 regularization weight for U
lambda_V = 0.001; % L2 regularization weight for v
T = 100; % No of gradient descent iterations
epsilon = 0.0001; % Convergence criterion for Gradient descent
eta = 0.01; %step size for gradient descent
q = 50; % No of anchor points

% Compute an Incomplete SVD of the observed values for the distance metric
[U_I,s_I,V_I] = svds(M,r); % Do a rank "r" approximate of the matrix M
normU_I = sum(U_I.^2,2).^0.5; % Norm of the user singular vectors
normV_I = sum(V_I.^2,2).^0.5; % Norm of the movie singular vectors

% Now pick a bunch of "q" random anchor points and compute the local low
% rank approximation at these points
%clearvars vect mapping IC uniq_userID
%U_I = sparse(U_I);
%V_I = sparse(V_I);

initV = (s_I*V_I')';
%for k = 1:q
%    U_k = initU;
%    V_k = V_I;
%end
%clearvars initU

for k = 1:q
    randID = round(rand(1)*n_train); % Get a random index from the training dataset  
    mov_r = movieID_All(rperm(randID)); % Column index of M
    user_r = userID_All(rperm(randID)); % Row index of M
    
    % Compute distance to every other user using epanechnikov kernel &
    % distance as the angle between the rows of U
    % (arccos(dot(U_i,U_j)/||U_i||||U_j||))
    d_user = acos( (U_I*U_I(user_r,:)') ./ (normU_I*normU_I(user_r)) );
    d_validuser = mapping(d_user < h1);
    K_user = (1 - d_user.^2).*(d_user < h1);
    %d_invaliduser = d_user >= h1;
    
    % Compute distance to every other movie using epanechnikov kernel &
    % distance as the angle between the rows of V
    % (arccos(dot(V_i,V_j)/||V_i||||V_j||))
    d_mov = acos( (V_I*V_I(mov_r,:)') ./ (normV_I*normV_I(mov_r)) );
    d_validmovie = moviemapping(d_mov < h2);
    K_mov = (1 - d_mov.^2).*(d_mov < h2);
    %d_invalidmovie = d_mov >= h2;
    
    kernel_union = ismember(userID_Train,d_validuser) & ismember(movieID_Train,d_validmovie);
    userID_All_filt = userID_Train(kernel_union);
    movieID_All_filt = movieID_Train(kernel_union);
    valid_indices = sub2ind(size(M),userID_All_filt,movieID_All_filt);
    
    clearvars d_user d_mov
    
    U_k = U_I;
    V_k = initV;
    
    loss_function_prev = inf;
    Mprime = bsxfun(@times,bsxfun(@times, M, K_user),K_mov');
    [U_k,s_k,V_k] = svds(Mprime,r);
    U_k = U_k*s_k;
    
    sumTerm = sum(bsxfun(@times,K_user(userID_All_filt),U_k(userID_All_filt,:)).*(bsxfun(@times,K_mov(movieID_All_filt),V_k(movieID_All_filt,:))),2)' - Mprime(valid_indices);
    %sumTerm = sumTerm ./ numel(sumTerm);
    tempmat = zeros(numel(userID_Train),1);
    
    % Now do gradient descent
    for iteration = 1:T
        fprintf('%d\n',iteration);
        
        
        %{
        prediction = U_k(user_r,:)*V_k(mov_r,:)';

        error = prediction - M(user_r,mov_r);
        delUVT_by_delU = zeros(size(U_I));
        delUVT_by_delV = zeros(size(V_I));
        delUVT_by_delU(user_r,:) = V_k(mov_r,:);
        delUVT_by_delV(mov_r,:) = U_k(user_r,:);
        
        U_k = U_k - eta*(lambda_U*U_k +  error*delUVT_by_delU);
        
        V_k = V_k - eta*(lambda_V*V_k +  error*delUVT_by_delV);
        %}
        
        %UVTransMinusM = (U_k*V_k' - M);
        %UVTransMinusM = bsxfun(@times,UVTransMinusM,K_user); %multiply ith row by K_user(i)
        %UVTransMinusM = bsxfun(@times,UVTransMinusM',K_mov)'; %multiply ith column by K_mov(i)
        
        tempmat(kernel_union) = sumTerm;
        tempmat(~kernel_union) = 0;
        UVTransMinusM = sparse(userID_Train,movieID_Train,tempmat);
        fprintf('Here1 \n');
%         
        U_k = U_k - eta*2*(lambda_U*U_k + UVTransMinusM*V_k);
        V_k = V_k - eta*2*(lambda_V*V_k + (U_k'*UVTransMinusM)');
%          
        fprintf('Here2 \n');

        %sumTerm = sum(U_k(userID_All_filt,:).*V_k(movieID_All_filt,:),2)' - M(valid_indices);
        sumTerm = sum(bsxfun(@times,K_user(userID_All_filt),U_k(userID_All_filt,:)).*(bsxfun(@times,K_mov(movieID_All_filt),V_k(movieID_All_filt,:))),2)' - Mprime(valid_indices);
        %sumTerm = sumTerm ./ numel(sumTerm);


        fprintf('Here2.5 \n');
        loss_function_new = lambda_U * norm(U_k, 'fro') + lambda_V * norm(V_k, 'fro') + sum(sumTerm.^2);
        fprintf('Here3 \n');

        if abs(loss_function_prev - loss_function_new) < epsilon
            fprintf('converged\n');
            break;
        else 
            loss_function_prev = loss_function_new;
        end
    end
end