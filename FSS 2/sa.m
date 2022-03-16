clc;
clear;
close all;

%% Problem Definition

data=LoadData();

nf=30;   % Desired Number of Selected Features

CostFunction=@(q) FeatureSelectionCost(q,nf,data);    % Cost Function

%% SA Parameters

MaxIt=10;     % Maximum Number of Iterations

MaxSubIt=5;    % Maximum Number of Sub-iterations

T0=10;         % Initial Temp.

alpha=0.99;     % Temp. Reduction Rate

%% Initialization

% Create and Evaluate Initial Solution
sol.Position=CreateRandomSolution(data);
[sol.Cost, sol.Out]=CostFunction(sol.Position);

% Initialize Best Solution Ever Found
BestSol=sol;

% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);

% Intialize Temp.
T=T0;

%% SA Main Loop

for it=1:MaxIt
    
    for subit=1:MaxSubIt
        
        % Create and Evaluate New Solution
        newsol.Position=CreateNeighbor(sol.Position);
        [newsol.Cost, newsol.Out]=CostFunction(newsol.Position);
        
        if newsol.Cost<=sol.Cost % If NEWSOL is better than SOL
            sol=newsol;
            
        else % If NEWSOL is NOT better than SOL
            
            DELTA=(newsol.Cost-sol.Cost)/sol.Cost;
            
            P=exp(-DELTA/T);
            if rand<=P
                sol=newsol;
            end
            
        end
        
        % Update Best Solution Ever Found
        if sol.Cost<=BestSol.Cost
            BestSol=sol;
        end
    
    end
    
    % Store Best Cost Ever Found
    BestCost(it)=BestSol.Cost;
    
    % Display Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    % Update Temp.
    T=alpha*T;
        
end

%% Results

figure;
plot(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');

