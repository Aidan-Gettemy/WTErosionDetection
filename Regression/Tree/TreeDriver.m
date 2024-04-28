clc;close;clear;
%% Designate Folder for plot results
% Subfolder for plots
plotID = "Evaluation/";
mkdir("Evaluation/")

%% Load in Data Inputs and Outputs 

InTraintableID = "../../InTable_Train_1700.txt";
InTesttableID = "../../InTable_Test_1700.txt";

OutTraintableID = "../../OutTable_Train_1700.txt";
OutTesttableID = "../../OutTable_Test_1700.txt";

In_train = readtable(InTraintableID);

In_test = readtable(InTesttableID);

Out_train = readtable(OutTraintableID);

Out_test = readtable(OutTesttableID);

% Don't predict the class
Out_train(:,19) = []; 
Out_test(:,19) = [];

InNames = In_train.Properties.VariableNames;
OutNames = Out_train.Properties.VariableNames;

% Now, assemble a matrix of the selected predictors for each output
Importances = zeros(numel(OutNames),numel(InNames));
iter = 1;
for j = 1:3
    for i = 1:6
        predimpID = "../../FeatureSelection/Blade"+num2str(j)+...
            "Region"+num2str(i)+"Regression_ranked_pred_imp.txt";
        Importances(iter,:)=readmatrix(predimpID);
        iter = iter+1;
    end
end


%% Set up the Model/Train the Model/Load Model

% Hold the predictions to use for plotting
Predicted_Train = zeros(numel(Out_train(:,1)),numel(OutNames));
Predicted_Test = zeros(numel(Out_test(:,1)),numel(OutNames));

Scores = zeros(numel(OutNames),6);

for i = 1:(numel(OutNames))
    disp(OutNames{i})
    OutName = OutNames{i};
    % Mdl = fitrtree(In_train(:,Importances(i,1:20)),Out_train(:,i).Variables,...
    % 'OptimizeHyperparameters','auto',...
    % 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    % 'expected-improvement-plus'));
    loadID = OutName+"rtree.mat";
    Mdl = load(loadID,'Mdl');Mdl = Mdl.Mdl;
    % Predict
    Out_Train_predict = Mdl.predict(In_train(:,Importances(i,1:20)));
    Out_Test_predict = Mdl.predict(In_test(:,Importances(i,1:20)));

    Predicted_Train(:,i) = Out_Train_predict;
    Predicted_Test(:,i) = Out_Test_predict;
    
    % Evaluate the Model

    % Training
    Scores(i,[1,3,5]) = CalcCoefs(Out_Train_predict,Out_train(:,i).Variables);
    % Testing 
    Scores(i,[2,4,6]) = CalcCoefs(Out_Test_predict,Out_test(:,i).Variables);
    
    % Save the model
    % saveID = OutName+"rtree.mat";
    % save(saveID,'Mdl');

end
%% Generate Plots and Save information

ScoreTable = array2table(Scores,'VariableNames',{'R_train','R_test','num_train',...
    'num_test','mse_train','mse_test'});
writetable(ScoreTable,'Scoretable.txt')

%% We will make all of the Plots as follows:

for i = 1:(numel(OutNames))
    name = OutNames{i};
    saveID = plotID+name;
    Predict = {Predicted_Train(:,i),Predicted_Test(:,i)};
    Actual = {Out_train(:,i).Variables, Out_test(:,i).Variables};
    scs = Scores(i,:);

    stat1 = makeScatter(name,saveID,Predict,Actual,scs);
    stat2 = makeQQ(name,saveID,Predict,Actual,scs);
end
%% Functions
% Calculate Metrics
function results = CalcCoefs(Predicted,Actual)
    % Inputs: Predicted vector and Actual vector
    % Outputs: 
    %           R: Correlation coefficient
    %           num; length of Predicted
    %           mse; mean square error

    R_1 = corrcoef(Predicted,Actual);
    R = R_1(1,2);

    num = numel(Predicted);

    mse = sum((Predicted - Actual).^2)/num;
    
    results = [R, num, mse];
end

function status = makeScatter(name,saveID,Predict,Actual,scores)
    figure
    hold on
    lgd{1} = ['1:1'];
    bds = [min(Predict{1}),max(Predict{1})];
    % set the colors
    cll = "#000080";
    ctr = "#ff8000";
    cte = "#009900";

    plot(bds,bds,'Color',cll,'LineWidth',10)

    % Plot Training
    scatter(Actual{1},Predict{1},20,'filled','MarkerFaceColor',ctr)
    % Plot Testing
    scatter(Actual{2},Predict{2},20,'filled','MarkerFaceColor',cte)
    % Add the legend
    lgd{2} = ['Training Data (R ' num2str(scores(1)) ', # ' num2str(scores(3)) ')'];
    lgd{3} = ['Testing Data (R ' num2str(scores(2)) ', # ' num2str(scores(4)) ')'];
    grid on
    legend(lgd,'Location','southeast','FontSize',10);
    xlabel("Actual",FontSize=15)
    ylabel("Estimated",FontSize=15)
    title("Tree Regression: "+name, 'FontSize',20);
    set(gca,'FontSize',16);
    set(gca,'LineWidth',2);
    saveID = saveID+"scatter.png";
    print('-dpng',saveID)
    status = "Scatter Plot for "+name+" is done."
    hold off
end

function status = makeQQ(name,saveID,Predict,Actual,scores)
    pvec=[0:5:15 25:25:75 85:5:100] ;
    qq_act_train=prctile(Actual{1},pvec);
    qq_pred_train=prctile(Predict{1},pvec);

    qq_act_test=prctile(Actual{2},pvec);
    qq_pred_test=prctile(Predict{2},pvec);

    bds = [min(Predict{1})*.1*(-1),max(Predict{1})+.2];
    figure
    cll = "#000080";
    ctr = "#ff8000";
    cte = "#009900";

    plot(bds,bds,'Color',cll,'LineWidth',7)
    hold on
    scatter(qq_act_train,qq_pred_train,150,'+','MarkerFaceColor',ctr,...
        'MarkerEdgeColor',ctr,'LineWidth',5)
    scatter(qq_act_test,qq_pred_test,150,'x','MarkerFaceColor',cte,...
        'MarkerEdgeColor',cte,'LineWidth',5)
    hold off
    del = .1*(bds(2)-bds(1));
    for i=1:length(pvec)
       text(...
           qq_act_train(i),qq_pred_train(i)+del,...
           num2str(pvec(i)),...
           'Color',ctr,...
           'FontSize',15,...
           'HorizontalAlignment','center',...
           'VerticalAlignment','top'...
           )
       text(...
           qq_act_test(i),qq_pred_test(i)-del,...
           num2str(pvec(i)),...
           'Color',cte,...
           'FontSize',15,...
           'HorizontalAlignment','center',...
           'VerticalAlignment','bottom'...
           )
    end
    
    set(gca,'LineWidth',1)
    set(gca,'TickDir','out')
    set(gca,'FontSize',16)                    
    xlabel('Data Quantiles','FontSize',15)
    ylabel('Estimate Quantiles','FontSize',15)
    title("Tree Regression: "+name ,'FontSize',20)
    xlim([bds(1)-del,bds(2)+del])
    ylim([bds(1)-del,bds(2)+del])
    grid on
    lgd{1} = ['1:1'];
    lgd{2} = ['Training Data (R ' num2str(scores(1)) ', # ' num2str(scores(3)) ')'];
    lgd{3} = ['Testing Data (R ' num2str(scores(2)) ', # ' num2str(scores(4)) ')'];
    legend(lgd,'Location','southeast','FontSize',5)
    saveID = saveID+"QQ.png";
    print('-dpng',saveID)
    status = "QQ Plot for "+name+" is done."
end
